"""Score one predictions file on the three official scopes and write a side-by-side summary.

    python scripts/run_eval.py PREDICTIONS --run-name NAME

Writes reports/NAME/ (full holdout), reports/NAME_no_near_dups/ (minus the MinHash near-copies),
reports/NAME_no_twins/ (minus the TF-IDF paraphrase twins), and reports/NAME/summary.md.
"""
import argparse
import sys
from pathlib import Path

from score import REPORTS, ROOT, ValidationError, headline_rows, score_run

SCOPES = [
    ("", None, "full"),
    ("_no_near_dups", REPORTS / "near_dup_holdout_ids.txt", "no near-dups"),
    ("_no_twins", REPORTS / "twin_holdout_ids.txt", "no twins"),
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def render_summary(run_name: str, payloads: list[dict]) -> str:
    # The excluded scopes live under "excluded"; their "full" block repeats the first run.
    scopes = [payloads[0]["full"]] + [p["excluded"] for p in payloads[1:]]
    meta = payloads[0]["meta"]
    head = " | ".join(label for _, _, label in SCOPES)
    L = [f"# Summary: {run_name}", "",
         f"- predictions: `{meta['predictions_path']}`",
         f"- sha256: `{meta['predictions_sha256']}`",
         f"- git commit: {meta['git_commit']}",
         f"- scored at: {meta['scored_at']}", ""]
    for (suffix, ids, label), p in zip(SCOPES, payloads):
        excl = f", excluding {p['meta']['excluded_ids']} IDs in `{rel(ids)}`" if ids else ""
        L.append(f"- {label}: [reports/{run_name}{suffix}/metrics.md](../{run_name}{suffix}/metrics.md){excl}")
    L += ["", "Rates are value (numerator/denominator). ECE uses 10 equal-width bins on the confidence "
          "of the predicted label.", "",
          f"| metric | {head} |", "|---|" + "---|" * len(SCOPES),
          "| tickets scored | " + " | ".join(str(s["n"]) for s in scopes) + " |"]
    rows = [headline_rows(s) for s in scopes]
    for cells in zip(*rows):
        L.append(f"| {cells[0][0]} | " + " | ".join(v for _, v in cells) + " |")
    lat = payloads[0]["full"].get("latency_ms")
    if lat and lat["n"]:
        L += ["", f"Latency, full holdout: p50 {lat['p50']:.0f} ms, p95 {lat['p95']:.0f} ms, max {lat['max']:.0f} ms "
              f"(n {lat['n']})."]
    return "\n".join(L) + "\n"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()

    payloads = []
    try:
        for suffix, ids, label in SCOPES:
            payloads.append(score_run(args.predictions, args.run_name + suffix, ids))
            print(f"scored {label}: reports/{args.run_name}{suffix}/")
    except ValidationError as e:
        print(e, file=sys.stderr)
        return 2

    shas = {p["meta"]["predictions_sha256"] for p in payloads}
    if len(shas) != 1:
        print(f"predictions file changed between scopes: {sorted(shas)}", file=sys.stderr)
        return 1
    out = REPORTS / args.run_name / "summary.md"
    text = render_summary(args.run_name, payloads)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
