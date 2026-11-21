"""How often each reviewed action is supported by KB answers in the fit partition.

Writes reports/d4/action_mining.md with support rates per answer language and example
evidence sentences, so the action catalogue can be checked against the data.
"""
from __future__ import annotations

import random

from _common import REPORTS, guess_language, load_kb

from sentineldesk.safety.actions import ACTIONS
from sentineldesk.safety.pii import mask_pii
from sentineldesk.safety.text import sentences


def main() -> None:
    kb = load_kb()
    fit = kb[kb["partition"] == "fit"]
    rng = random.Random(42)
    rows = []
    counts = {a.id: {"en": 0, "de": 0} for a in ACTIONS}
    examples = {a.id: {"en": [], "de": []} for a in ACTIONS}
    totals = {"en": 0, "de": 0}
    any_support = {"en": 0, "de": 0}
    for answer in fit["answer"]:
        answer = mask_pii(answer).text
        lang = guess_language(answer)
        totals[lang] += 1
        sents = sentences(answer)
        hit_any = False
        for a in ACTIONS:
            match = next((s for s in sents if a.supports(s)), None)
            if match:
                hit_any = True
                counts[a.id][lang] += 1
                examples[a.id][lang].append(match)
        any_support[lang] += hit_any

    lines = [
        "# Action support in KB answers (fit partition)",
        "",
        f"Answers: {totals['en']:,} English, {totals['de']:,} German (language guessed from function words).",
        f"At least one action supported: English {any_support['en'] / max(totals['en'], 1):.1%}, "
        f"German {any_support['de'] / max(totals['de'], 1):.1%}.",
        "",
        "| action | English answers | German answers |",
        "|---|---|---|",
    ]
    for a in ACTIONS:
        en, de = counts[a.id]["en"], counts[a.id]["de"]
        lines.append(f"| `{a.id}` | {en:,} ({en / max(totals['en'], 1):.1%}) | "
                     f"{de:,} ({de / max(totals['de'], 1):.1%}) |")
    lines += ["", "## Example evidence sentences (seeded sample)", ""]
    for a in ACTIONS:
        lines.append(f"### `{a.id}`")
        lines.append(f"Reply text: EN \"{a.en}\" · DE \"{a.de}\"")
        lines.append("")
        for lang in ("en", "de"):
            pool = examples[a.id][lang]
            for s in rng.sample(pool, min(3, len(pool))):
                lines.append(f"- ({lang}) {s}")
        lines.append("")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "action_mining.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
