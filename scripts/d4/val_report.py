"""Gate reports on the KB validation partition (KB data and KB labels only).

Writes reports/d4/rule_rates.md, reports/d4/outage_rules.md and reports/d4/thresholds.md.
Triage and retrieval come from the dev-only stand-ins (uncalibrated triage fit on `fit`,
retrieval over `fit` only), so every number here is provisional until D2's calibrated
probabilities and D3's retrieval are rerun through the same script.

Selection rules:
- type: the lowest confidence threshold with at least 80% accuracy at or above it (min 0.45)
- queue and priority: the lowest pair where both labels are right for at least 80% of the
  tickets that end with a validated draft
- evidence: the lowest similarity where the best precedent's queue AND type agree with
  the ticket's labels for at least 80% of val tickets at or above it
"""
from __future__ import annotations

import numpy as np
from _common import REPORTS, load_kb
from dev_standins import RetrievalStandIn, load_triage

from sentineldesk.safety import contracts as c
from sentineldesk.safety.config import SafetyConfig
from sentineldesk.safety.drafting import draft_template
from sentineldesk.safety.gate import gate
from sentineldesk.safety.patterns import authorization_terms, instruction_override, integration_request, outage_impact
from sentineldesk.safety.validation import validate

HEADS = ("type", "queue", "priority")
HEAD_TARGET, EVIDENCE_TARGET = 0.80, 0.80
GRID = np.round(np.arange(0.20, 0.91, 0.05), 2)
SIM_GRID = np.round(np.arange(0.10, 0.61, 0.02), 2)
PAIR_GRID = (0.45, 0.55, 0.65, 0.75)


def pct(x: float) -> str:
    return f"{x:.1%}"


def main() -> None:
    kb = load_kb()
    fit, val = kb[kb["partition"] == "fit"], kb[kb["partition"] == "val"].reset_index(drop=True)
    labels = kb.set_index("ticket_id")[["type", "queue", "priority"]]
    tri_model, retriever = load_triage(), RetrievalStandIn(fit)
    triages = tri_model.predict(val["subject"], val["body"])
    precs = [retriever.search(s, b, k=5) for s, b in zip(val["subject"], val["body"])]
    n = len(val)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ rule trigger rates
    text = (val["subject"] + "\n" + val["body"]).tolist()
    joint = ((val["type"] == "Incident") & (val["priority"] == "high")).to_numpy()
    rates = {
        "Rule 2 wording (outage + scope + critical service)": [bool(outage_impact(t)) for t in text],
        "Rule 3 instruction override": [bool(instruction_override(t)) for t in text],
        "Rule 5 authorization required": [bool(authorization_terms(t)) for t in text],
        "Rule 8 integration request": [bool(integration_request(t)) for t in text],
    }
    L = ["# Gate rule trigger rates (KB validation partition)", "", f"{n:,} tickets.", "",
         "| rule | fires | share |", "|---|---|---|"]
    L += [f"| {k} | {sum(v):,} | {pct(sum(v) / n)} |" for k, v in rates.items()]
    hits = np.array(rates["Rule 2 wording (outage + scope + critical service)"])
    L += ["", f"Rule 2 wording hits that are true Incident + high: {int((hits & joint).sum())} of {int(hits.sum())}."]
    (REPORTS / "rule_rates.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ outage rule variants
    pred_inc = np.array([t.type.label == "Incident" for t in triages])
    pred_high = np.array([t.priority.label == "high" for t in triages])
    p_inc = np.array([t.type.probabilities.get("Incident", 0.0) for t in triages])
    p_high = np.array([t.priority.probabilities.get("high", 0.0) for t in triages])

    def row(name, mask):
        tp = int((mask & joint).sum())
        return (f"| {name} | {tp}/{int(joint.sum())} = {pct(tp / joint.sum())} | {int(mask.sum())} = {pct(mask.mean())} "
                f"| {pct(tp / max(mask.sum(), 1))} |")

    cfg = SafetyConfig()
    a, b = cfg.outage_pair
    L = ["# Outage rule variants (KB validation partition, provisional stand-in triage)", "",
         f"True Incident + high: {int(joint.sum())} of {n} ({pct(joint.mean())}).", "",
         "| variant | recall of true Incident + high | sent to on-call | precision |", "|---|---|---|---|",
         row("Predicted Incident + high (brief, rule 2)", pred_inc & pred_high),
         row("Predicted high alone", pred_high),
         row(f"Rule 2 + rule 6 at P(Incident) >= {a}, P(high) >= {b}", (pred_inc & pred_high) | ((p_inc >= a) & (p_high >= b))),
         row("Rule 2 + rule 6 + rule 2 wording", (pred_inc & pred_high) | ((p_inc >= a) & (p_high >= b)) | hits)]
    best = None
    for ta in np.round(np.arange(0.30, 0.71, 0.05), 2):
        for tb in np.round(np.arange(0.30, 0.71, 0.05), 2):
            m = (pred_inc & pred_high) | ((p_inc >= ta) & (p_high >= tb))
            rec, prec = (m & joint).sum() / joint.sum(), (m & joint).sum() / max(m.sum(), 1)
            if rec >= 0.90 and (best is None or prec > best[2]):
                best = (ta, tb, prec, rec, m.mean())
    L += ["", "Plan search for the rule 6 pair (0.30–0.70, step 0.05, highest precision with combined recall >= 90%): "
          + ("none reaches 90%." if best is None else
             f"P(Incident) >= {best[0]}, P(high) >= {best[1]}: recall {pct(best[3])}, precision {pct(best[2])}, "
             f"on-call {pct(best[4])}."),
          "D2 selects the release pair on its calibrated probabilities; this table shows the trade-off."]
    (REPORTS / "outage_rules.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ evidence threshold
    best_score = np.array([p[0].score if p else 0.0 for p in precs])
    agree = np.array([bool(p) and labels.loc[p[0].ticket_id, "queue"] == q and labels.loc[p[0].ticket_id, "type"] == ty
                      for p, q, ty in zip(precs, val["queue"], val["type"])])
    L = ["# Thresholds (KB validation partition, provisional stand-ins)", "",
         f"## Evidence (rule 8)", "",
         f"Lowest best-precedent similarity where the precedent's queue and type both match the ticket for at "
         f"least {pct(EVIDENCE_TARGET)} of tickets at or above it.", "",
         "| similarity | agreement at or above | share at or above |", "|---|---|---|"]
    evidence = None
    for sim in SIM_GRID:
        m = best_score >= sim
        if m.sum() < 20:
            break
        L.append(f"| {sim:.2f} | {pct(agree[m].mean())} | {pct(m.mean())} |")
        if evidence is None and agree[m].mean() >= EVIDENCE_TARGET:
            evidence = float(sim)
    evidence = evidence if evidence is not None else float(SIM_GRID[-1])

    # ------------------------------------------------------------ confidence thresholds (rule 7)
    def run(conf: SafetyConfig):
        statuses, routed_right = [], []
        for i in range(n):
            masked = c.MaskedTicket(subject=val.at[i, "subject"], body=val.at[i, "body"])
            lang = "de" if val.at[i, "language"] == "de" else "en"  # KB label, adequate for a rate estimate
            status = gate(masked, lang, triages[i], precs[i], config=conf).status
            if status == c.DRAFT_READY:
                dr = draft_template(masked, lang, precs[i], config=conf)
                if dr.abstain or not validate(dr, lang, precs[i], ticket=masked, config=conf).valid:
                    status = "draft_failed"
                else:
                    routed_right.append(triages[i].queue.label == val.at[i, "queue"]
                                        and triages[i].priority.label == val.at[i, "priority"])
            statuses.append(status)
        return statuses, routed_right

    type_conf = np.array([t.type.confidence for t in triages])
    type_ok = np.array([t.type.label == y for t, y in zip(triages, val["type"])])
    type_thr = max(0.45, next((float(t) for t in GRID if type_ok[type_conf >= t].mean() >= HEAD_TARGET), 0.90))
    L += ["", "## Confidence (rule 7)", "",
          f"Type: lowest threshold with at least {pct(HEAD_TARGET)} accuracy at or above it, never below the plan's "
          f"starting 0.45: {type_thr:.2f} (overall type accuracy {pct(type_ok.mean())}).", "",
          f"Queue and priority: the lowest pair where queue and priority are both right for at least "
          f"{pct(HEAD_TARGET)} of the tickets that get a validated draft.", "",
          "| queue | priority | validated drafts | queue and priority right among them |", "|---|---|---|---|"]
    chosen = None
    for total, q, pr in sorted((q + pr, q, pr) for q in PAIR_GRID for pr in PAIR_GRID):
        conf = SafetyConfig(head_thresholds={"type": type_thr, "queue": q, "priority": pr},
                            evidence_min={"sparse": evidence, "dense": 0.80, "fixture": 0.0})
        statuses, right = run(conf)
        share, acc = statuses.count(c.DRAFT_READY) / n, (np.mean(right) if right else 0.0)
        L.append(f"| {q:.2f} | {pr:.2f} | {pct(share)} | {pct(acc)} |")
        if chosen is None and right and acc >= HEAD_TARGET:
            chosen = (q, pr, conf, statuses)
    if chosen is None:
        q = pr = float(PAIR_GRID[-1])
        conf = SafetyConfig(head_thresholds={"type": type_thr, "queue": q, "priority": pr},
                            evidence_min={"sparse": evidence, "dense": 0.80, "fixture": 0.0})
        chosen = (q, pr, conf, run(conf)[0])
    q, pr, _, tuned_statuses = chosen
    default_statuses, _ = run(SafetyConfig())

    L += ["", f"**Chosen:** type {type_thr:.2f}, queue {q:.2f}, priority {pr:.2f}, sparse evidence {evidence:.2f}.",
          "", "| outcome | plan defaults (0.45 each, evidence 0.30) | chosen |", "|---|---|---|"]
    for k in (c.DRAFT_READY, "draft_failed", c.REVIEW, c.ESCALATE, c.NEEDS_CLARIFICATION):
        L.append(f"| {k} | {pct(default_statuses.count(k) / n)} | {pct(tuned_statuses.count(k) / n)} |")
    L += ["", "draft_ready counts validated drafts only. No hand labels were available, so safety was not checked "
          "by a person; label agreement is a proxy. The drafts only ask for information that a cited past answer "
          "also asked for, so the worst case is an unhelpful draft, not an invented commitment."]
    (REPORTS / "thresholds.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    for f in ("rule_rates.md", "outage_rules.md", "thresholds.md"):
        print((REPORTS / f).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
