"""Shared text normalization for training, retrieval and serving.

The locked split keeps text raw on purpose: live tickets arrive raw, so every
consumer (classifier, retrieval, API) runs the same function on the same input.

    from scripts.text_normalize import normalize_text, ticket_text, find_placeholders
"""
from __future__ import annotations

import html
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

PLACEHOLDER_MODES = ("keep", "canonical", "remove")
CANONICAL_KINDS = ("name", "tel_num", "acc_num", "email", "url", "amount")

_BLOCK_TAGS = r"br|p|div|li|ul|ol|tr|td|th|table|tbody|thead|h[1-6]|hr|blockquote|pre|body|html|head|center"
_INLINE_TAGS = r"span|b|i|u|strong|em|font|a|sup|sub|small|code|img"
# Tag names only, so mask tokens such as <tel_num> or <name> never match here.
_BLOCK_TAG_RE = re.compile(rf"</?\s*(?:{_BLOCK_TAGS})\b[^<>]{{0,200}}?/?\s*>", re.I)
_INLINE_TAG_RE = re.compile(rf"</?\s*(?:{_INLINE_TAGS})\b[^<>]{{0,200}}?/?\s*>", re.I)

_HSPACE_RE = re.compile(r"[^\S\n]+")
_SPACE_AROUND_NL_RE = re.compile(r" *\n *")
_MANY_NL_RE = re.compile(r"\n{3,}")

# Keys are the token text lowercased with possessive prefixes and all
# non-alphanumerics dropped. Variants are the ones counted in kb_train answers
# and bodies (e.g. "[Tel Nummer]", "<Tel-Nummer>", "[Konto-Nr.]").
_PLACEHOLDER_KEYS = {
    "name": "name", "firstname": "name", "fullname": "name", "customername": "name",
    "customersname": "name", "usersname": "name",
    "telnum": "tel_num", "telnumber": "tel_num", "telnummer": "tel_num", "telnr": "tel_num",
    "telefonnummer": "tel_num", "phonenumber": "tel_num", "phone": "tel_num",
    "telephonenumber": "tel_num", "ntelnum": "tel_num", "autelnum": "tel_num",
    "accnum": "acc_num", "accnr": "acc_num", "accountnumber": "acc_num",
    "accountnummer": "acc_num", "kontonummer": "acc_num", "kontonr": "acc_num",
    "naccnum": "acc_num",
    "email": "email", "emailaddress": "email", "emailadresse": "email",
    "emailprotected": "email",
    "url": "url", "websiteurl": "url", "website": "url", "link": "url",
    "amount": "amount", "betrag": "amount",
}
_PLACEHOLDER_RE = re.compile(r"<([^<>\n]{1,30})>|\[([^\[\]\n]{1,30})\]")
_PLACEHOLDER_WITH_SPACE_RE = re.compile(r"[^\S\n]*(?:<([^<>\n]{1,30})>|\[([^\[\]\n]{1,30})\])")
_KEY_PREFIX_RE = re.compile(r"^(?:your|ihre?|deine?|the|customer'?s|user'?s)\s+")
_KEY_STRIP_RE = re.compile(r"[\W_]+")

_GREET = (
    r"(?:dear|hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening|day)"
    # gesehrte/gehörte/geachte are misspellings seen opening kb bodies (Gehörte 20, Geachte 32).
    r"|to\s+whom\s+it\s+may\s+concern|sehr\s+geehrte[rs]?|gesehrte[rs]?|geehrte[rs]?|gehörte[rs]?|geachte[rs]?"
    r"|liebe[rs]?|hallo|guten\s+(?:tag|morgen|abend)|moin|servus)"
)
# Up to 6 addressee words; a word that starts the actual message ("Hi I can't
# log in") ends the salutation match instead of being swallowed.
_ADDR_WORD = r"(?:\s+(?!(?:i|i'm|im|we|my|our|ich|wir|mein\w*|unser\w*|es|das|die)\b)[^\s,:;!?.]+)"
_ADDRESSEE = (
    r"(?:(?:to\s+)?(?:the\s+|our\s+)?(?:customer|client|technical|tech|it|product)?[\s-]?"
    r"(?:support|service|care|help\s?desk)(?:[\s-]?(?:team|department|abteilung))?"
    r"|kunden(?:dienst|service|support|betreuung)[\w-]*|support[\s-]?team)"
)
_SALUTATION_RE = re.compile(
    rf"^\s*(?:{_GREET}\b{_ADDR_WORD}{{0,6}}\s*(?:[,:;!.]|\n|$)|{_ADDRESSEE}\s*(?:[,:;!]|\n))",
    re.I,
)
_PLEASANTRY_RE = re.compile(
    r"^\s*(?:"
    r"(?:i\s+|we\s+)?(?:hope|trust)\s+(?:that\s+)?(?:this|my|our)\s+(?:message|email|e-mail|note|letter|mail)\s+"
    r"(?:finds|reaches|gets\s+to)\s+you(?:\s+(?:well|in\s+good\s+health|in\s+good\s+spirits|all\s+well|safe\s+and\s+well|doing\s+well))?"
    r"(?:\s+and\s+in\s+good\s+\w+)?"
    r"|(?:i\s+|we\s+)?hope\s+(?:you\s+are|you're|you\s+have\s+been|all\s+is)\s+(?:doing\s+)?(?:well|good|great|fine)(?:\s+today)?"
    r"|ich\s+hoffe,?\s+(?:diese|meine|die)\s+(?:nachricht|e-mail|email|mail)\s+(?:erreicht|findet|trifft)\s+sie"
    r"(?:\s+(?:wohl\w*|gut|bei\s+bester\s+gesundheit|in\s+guter\s+gesundheit|gesund))?"
    r"|ich\s+hoffe,?\s+(?:es\s+geht\s+ihnen\s+gut|sie\s+befinden\s+sich\s+(?:gut|wohl\w*)"
    r"|sie\s+sind\s+(?:wohlauf|gesund|in\s+bester\s+gesundheit)|sie\s+erhalten\s+diese\s+nachricht)"
    r")\s*[.!]+",
    re.I,
)

_VALEDICTION = (
    r"(?:best\s+regards|kind\s+regards|warm\s+regards|warmest\s+regards|regards|best\s+wishes"
    r"|sincerely|yours\s+(?:sincerely|faithfully|truly)|respectfully|cheers|many\s+thanks"
    r"|thanks|thank\s+you|best"
    r"|mit\s+freundlichen\s+grüßen|freundliche\s+grüße|viele\s+grüße|beste\s+grüße|liebe\s+grüße"
    r"|herzliche\s+grüße|mit\s+besten\s+grüßen|mfg|hochachtungsvoll|gruß|vielen\s+dank)"
)
# Valediction at the start of a line/sentence followed only by a short tail
# (a name, a title, a placeholder) up to the end of the text.
_VALEDICTION_RE = re.compile(rf"(?:^|(?<=[.!?\n]))[^\S\n]*{_VALEDICTION}[^\S\n]*(?:[,.!:]|(?=\n)|$)", re.I)
_VALEDICTION_TAIL_MAX_WORDS = 8
# A signature tail is capitalized words or placeholders ("John Smith, IT Lead",
# "[Your Name]"); any other lowercase word means the message goes on.
_SIGNATURE_CONNECTORS = {"of", "and", "at", "for", "und", "von", "der", "die", "das", "bei", "&", "-"}

_THANK_OBJ = (
    r"(?:time|help|assistance|support|attention|patience|understanding|consideration|cooperation"
    r"|response|reply|guidance|efforts?)"
)
_THIS_MATTER = r"(?:\s+(?:in|on|with|regarding|to)\s+(?:this|the)\s+(?:matter|issue|request|problem))?"
_THANKS = (
    rf"(?:thank\s+you|thanks|many\s+thanks)(?:\s+(?:so|very)\s+much|\s+a\s+lot)?(?:\s+(?:again|in\s+advance))?"
    rf"(?:\s+for\s+(?:your|the|all\s+(?:of\s+)?your)?\s*(?:kind\s+|prompt\s+|quick\s+|continued\s+|valuable\s+)?{_THANK_OBJ}"
    rf"(?:\s+(?:and|&)\s+(?:your\s+)?{_THANK_OBJ})?{_THIS_MATTER})?"
    rf"(?:\s+in\s+advance)?"
)
_LOOK_FWD_TAIL = (
    r"\s+forward\s+to\s+"
    r"(?:hearing(?:\s+back)?(?:\s+from\s+you)?|your\s+(?:(?:prompt|quick|timely|swift|kind|soonest|detailed)\s+(?:and\s+)?){0,2}"
    r"(?:response|reply|feedback|assistance|support|guidance)|an?\s+(?:prompt\s+|quick\s+|swift\s+|timely\s+|positive\s+)?"
    r"(?:response|reply|resolution))(?:\s+from\s+you)?(?:\s+(?:soon|shortly|at\s+your\s+earliest\s+convenience))?"
)
_LOOK_FWD = rf"(?:(?:i|we)\s+(?:look|am\s+looking|are\s+looking)|looking){_LOOK_FWD_TAIL}"
_AND_LOOK_FWD = rf"(?:,?\s+and\s+(?:(?:i|we)\s+)?(?:look|am\s+looking|are\s+looking){_LOOK_FWD_TAIL})?"
_DE_THANKS_OBJ = r"(?:zeit|hilfe|unterstützung|aufmerksamkeit|geduld|verständnis|mühe|rückmeldung|bemühungen|kooperation)"
_DE_THANKS = (
    rf"(?:(?:vielen|herzlichen|besten)\s+dank|danke(?:\s+schön|\s+sehr)?)(?:\s+im\s+voraus)?"
    rf"(?:\s+für\s+(?:ihre|die|ihr|ihren)\s+(?:\w+\s+)?{_DE_THANKS_OBJ}(?:\s+und\s+(?:ihre\s+|die\s+)?{_DE_THANKS_OBJ})?"
    rf"(?:\s+(?:in|bei)\s+dieser\s+angelegenheit)?)?"
    rf"(?:\s+im\s+voraus)?"
)
_DE_LOOK_FWD = (
    r"(?:ich\s+freue\s+mich|wir\s+freuen\s+uns)\s+(?:auf\s+(?:ihre|eine)\s+(?:baldige\s+|schnelle\s+|zeitnahe\s+|positive\s+)?"
    r"(?:rückmeldung|antwort)|darauf,?\s+(?:bald\s+)?von\s+ihnen\s+zu\s+hören"
    r"|darauf,?\s+(?:bald\s+)?(?:eine|ihre)\s+(?:rückmeldung|antwort)\s+zu\s+(?:erhalten|bekommen))"
    r"|ich\s+hoffe,?\s+bald\s+von\s+ihnen\s+zu\s+hören"
)
_CLOSING_RE = re.compile(
    rf"(?:{_THANKS}{_AND_LOOK_FWD}|{_LOOK_FWD}"
    r"|(?:your|any)\s+(?:help|assistance|support)\s+(?:would\s+be|is|will\s+be)\s+(?:greatly\s+|much\s+|highly\s+)?appreciated"
    r"|(?:i|we)\s+(?:would\s+)?(?:greatly\s+|really\s+)?appreciate\s+(?:your|any)\s+(?:prompt\s+|quick\s+|timely\s+)?"
    rf"(?:help|assistance|support|attention|response){_THIS_MATTER}{_AND_LOOK_FWD}"
    r"|(?:i|we)\s+(?:would\s+)?(?:greatly\s+|really\s+)?appreciate\s+it\s+if\s+you\s+could\s+look\s+into\s+this\s+(?:matter|issue)"
    r"(?:\s+(?:urgently|promptly|as\s+soon\s+as\s+possible))?(?:\s+and\s+provide\s+an?\s+(?:update|resolution|solution)"
    r"(?:\s+(?:soon|as\s+soon\s+as\s+possible|at\s+your\s+earliest\s+convenience))?)?"
    r"|(?:i|we)\s+kindly\s+request\s+your\s+(?:assistance|help|support)\s+in\s+resolving\s+this\s+(?:matter|issue)"
    r"(?:\s+(?:as\s+soon\s+as\s+possible|promptly|urgently))?"
    r"|please\s+keep\s+(?:me|us)\s+(?:informed|updated|posted)"
    r"|please\s+let\s+(?:me|us)\s+know(?:\s+if\s+(?:you\s+(?:need|require)|there\s+is)\s+(?:any\s+)?(?:additional|further|more)\s+"
    r"(?:information|details)(?:\s+(?:you\s+need|needed|required))?(?:\s+from\s+(?:me|us))?"
    r"(?:\s+to\s+resolve\s+(?:the|this)\s+(?:issue|problem|matter))?)?"
    r"|bitte\s+lassen\s+sie\s+(?:mich|uns)\s+wissen,?\s+(?:falls|wenn|ob)\s+sie\s+(?:weitere|zusätzliche)\s+(?:informationen|details)\s+benötigen"
    rf"|{_DE_THANKS}(?:,?\s+(?:und\s+)?{_DE_LOOK_FWD})?|{_DE_LOOK_FWD}"
    rf"|{_VALEDICTION})\s*[.!,]*",
    re.I,
)
_SEGMENT_BOUNDARY_RE = re.compile(r"[.!?](?=\s)|\n")
_HAS_WORD_RE = re.compile(r"\w")


def _unescape(text: str) -> str:
    # Repeat so double-escaped input ("&amp;nbsp;") settles in one call; keeps
    # normalize_text idempotent.
    for _ in range(3):
        new = html.unescape(text)
        if new == text:
            break
        text = new
    return text


def _strip_html(text: str) -> str:
    return _INLINE_TAG_RE.sub("", _BLOCK_TAG_RE.sub("\n", text))


def _placeholder_kind(token_inner: str) -> str | None:
    key = _KEY_PREFIX_RE.sub("", token_inner.strip().lower())
    return _PLACEHOLDER_KEYS.get(_KEY_STRIP_RE.sub("", key))


def _apply_placeholders(text: str, mode: str) -> str:
    if mode == "keep":
        return text

    def repl(m: re.Match) -> str:
        kind = _placeholder_kind(m.group(1) or m.group(2))
        if kind is None:
            return m.group(0)
        return f"<{kind}>" if mode == "canonical" else ""

    # Removal also eats the space before the token so "at <tel_num>." becomes "at." not "at .".
    pattern = _PLACEHOLDER_RE if mode == "canonical" else _PLACEHOLDER_WITH_SPACE_RE
    # Loop because removing an inner token can expose an outer one
    # ("[Account-Nummer: <acc_num>]" seen once in kb answers).
    for _ in range(3):
        new = pattern.sub(repl, text)
        if new == text:
            break
        text = new
    return text


def _tidy_whitespace(text: str) -> str:
    text = _HSPACE_RE.sub(" ", text)
    text = _SPACE_AROUND_NL_RE.sub("\n", text)
    text = _MANY_NL_RE.sub("\n\n", text)
    return text.strip()


def _strip_leading_counted(text: str) -> tuple[str, int, int]:
    n_sal = n_plea = 0
    while True:
        if m := _SALUTATION_RE.match(text):
            n_sal += 1
        elif m := _PLEASANTRY_RE.match(text):
            n_plea += 1
        else:
            return text, n_sal, n_plea
        text = text[m.end():].lstrip(" \n,;:")


def _strip_leading(text: str) -> str:
    return _strip_leading_counted(text)[0]


def _last_segment_start(text: str) -> int:
    starts = [m.end() for m in _SEGMENT_BOUNDARY_RE.finditer(text)]
    return starts[-1] if starts else 0


def _is_name_only(segment: str) -> bool:
    m = _PLACEHOLDER_RE.fullmatch(segment.strip(" ,."))
    return bool(m) and _placeholder_kind(m.group(1) or m.group(2)) == "name"


def _is_signature(tail: str) -> bool:
    words = tail.split()
    if len(words) > _VALEDICTION_TAIL_MAX_WORDS or len(tail) > 100 or "?" in tail:
        return False
    return all(not w[0].islower() or w.lower() in _SIGNATURE_CONNECTORS for w in words)


def _strip_trailing(text: str) -> str:
    while text:
        cut = None
        for m in _VALEDICTION_RE.finditer(text):
            if _is_signature(text[m.end():]):
                cut = m.start()
                break
        if cut is None:
            start = _last_segment_start(text)
            segment = text[start:].strip()
            if _CLOSING_RE.fullmatch(segment) or _is_name_only(segment):
                cut = start
        if cut is None:
            return text
        text = text[:cut].rstrip(" \n,;:")
    return text


def _strip_greeting(text: str) -> str:
    return _tidy_whitespace(_strip_trailing(_strip_leading(text)))


def normalize_text(text: str | None, *, strip_greeting: bool = False, placeholders: str = "keep") -> str:
    if placeholders not in PLACEHOLDER_MODES:
        raise ValueError(f"placeholders must be one of {PLACEHOLDER_MODES}, got {placeholders!r}")
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = _unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = _strip_html(text)
    text = _apply_placeholders(text, placeholders)
    text = _tidy_whitespace(text)
    if strip_greeting and text:
        stripped = _strip_greeting(text)
        if _HAS_WORD_RE.search(stripped):
            return stripped
    return text


def ticket_text(subject: str | None, body: str | None, **kw) -> str:
    """Subject, blank line, body. The single model and retrieval input string.

    Greeting stripping applies to the body only: subjects are one line and a
    subject such as "Thank you" would otherwise be emptied.
    """
    subj_kw = {k: v for k, v in kw.items() if k != "strip_greeting"}
    subj = normalize_text(subject, **subj_kw)
    body_n = normalize_text(body, **kw)
    if not subj:
        return body_n
    if not body_n:
        return subj
    return f"{subj}\n\n{body_n}"


def find_placeholders(text: str | None) -> list[str]:
    """Canonical placeholder kinds present in text, sorted."""
    if not text:
        return []
    kinds = {_placeholder_kind(m.group(1) or m.group(2)) for m in _PLACEHOLDER_RE.finditer(text)}
    return sorted(k for k in kinds if k)


def _word_count(text: str) -> int:
    return len(text.split())


def _summarize(root: Path) -> str:
    import pandas as pd

    kb = pd.read_parquet(root / "data/processed/kb_train.parquet")
    ti = pd.read_parquet(root / "data/processed/test_inputs.parquet")
    frames = {"kb_train.body": kb["body"], "test_inputs.body": ti["body"], "kb_train.answer": kb["answer"],
              "kb_train.subject": kb["subject"], "test_inputs.subject": ti["subject"]}

    rules = {
        "html unescape": _unescape,
        "html tags": _strip_html,
        "unicode NFC": lambda t: unicodedata.normalize("NFC", t),
        "space/tab runs": lambda t: _HSPACE_RE.sub(" ", t),
        "3+ newlines": lambda t: _MANY_NL_RE.sub("\n\n", t),
        "outer whitespace": str.strip,
        "placeholders canonical": lambda t: _apply_placeholders(t, "canonical"),
        "full normalize (keep)": normalize_text,
    }
    lines = ["# Text normalization summary", "",
             "Generated by `python scripts/text_normalize.py`. Rows changed per rule, each rule applied alone to raw text.", ""]
    header = "| rule | " + " | ".join(frames) + " |"
    lines += [header, "|" + "---|" * (len(frames) + 1)]
    for name, fn in rules.items():
        counts = [int(sum(fn(t) != t for t in s)) for s in frames.values()]
        lines.append(f"| {name} | " + " | ".join(str(c) for c in counts) + " |")
    lines += ["", f"Rows: kb_train {len(kb)}, test_inputs {len(ti)}.", ""]

    lines += ["## Greeting strip (bodies)", "",
              "| set | rows | touched | left as is | salutation removed | pleasantry removed | sign-off removed | fell back (would be empty) | mean words removed (all rows) | mean words removed (touched) | empty after |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in (("kb_train", kb["body"]), ("test_inputs", ti["body"])):
        base = [normalize_text(t) for t in s]
        out = [normalize_text(t, strip_greeting=True) for t in s]
        counted = [_strip_leading_counted(b) for b in base]
        sal = sum(c[1] > 0 for c in counted)
        plea = sum(c[2] > 0 for c in counted)
        sign = sum(_strip_trailing(c[0]) != c[0] for c in counted)
        fallback = sum(not _HAS_WORD_RE.search(_strip_greeting(b)) for b in base if b)
        touched = [i for i, (b, o) in enumerate(zip(base, out)) if b != o]
        removed = [_word_count(b) - _word_count(o) for b, o in zip(base, out)]
        mean_all = sum(removed) / len(removed)
        mean_touched = sum(removed[i] for i in touched) / max(len(touched), 1)
        empty = sum(not o for o in out)
        lines.append(f"| {name} | {len(s)} | {len(touched)} | {len(s) - len(touched)} | {sal} | {plea} | {sign} | "
                     f"{fallback} | {mean_all:.2f} | {mean_touched:.2f} | {empty} |")
    short_kb = sum(_word_count(normalize_text(t, strip_greeting=True)) < 5 for t in kb["body"])
    short_ti = sum(_word_count(normalize_text(t, strip_greeting=True)) < 5 for t in ti["body"])
    lines += ["", f"Bodies under 5 words after stripping: kb_train {short_kb}, test_inputs {short_ti}.", ""]

    variants: Counter = Counter()
    kinds: Counter = Counter()
    for s in frames.values():
        for t in s:
            for m in _PLACEHOLDER_RE.finditer(t):
                kind = _placeholder_kind(m.group(1) or m.group(2))
                if kind:
                    variants[m.group(0)] += 1
                    kinds[kind] += 1
    lines += ["## Placeholder variants (all text columns)", "",
              f"{len(variants)} distinct variants, {sum(variants.values())} occurrences.", "",
              "| canonical | occurrences |", "|---|---|"]
    lines += [f"| `<{k}>` | {v} |" for k, v in kinds.most_common()]
    lines += ["", "| variant | occurrences |", "|---|---|"]
    lines += [f"| `{v}` | {c} |" for v, c in variants.most_common(40)]

    lines += ["", "## Examples (body, strip_greeting=True, placeholders=canonical)", ""]
    picks = _pick_examples(kb)
    for i, (lang, raw) in enumerate(picks, 1):
        after = normalize_text(raw, strip_greeting=True, placeholders="canonical")
        lines += [f"### {i}. {lang}", "", "Before:", "", "```", raw.strip(), "```", "", "After:", "", "```", after, "```", ""]
    return "\n".join(lines)


def _pick_examples(kb) -> list[tuple[str, str]]:
    want = [
        ("en", lambda t: "<br" in t and _SALUTATION_RE.match(t)),
        ("en", lambda t: "I hope this message" in t[:120] and "[Your Name]" in t),
        ("en", lambda t: ("<tel_num>" in t or "[Name]" in t) and _SALUTATION_RE.match(t)),
        ("de", lambda t: "ich hoffe, diese Nachricht" in t[:120]),
        ("de", lambda t: "Mit freundlichen Grüßen" in t),
    ]
    picks, used = [], set()
    for lang, pred in want:
        for idx, row in kb[kb["language"] == lang].iterrows():
            if idx not in used and len(row["body"]) < 700 and pred(row["body"]):
                picks.append((lang, row["body"]))
                used.add(idx)
                break
    return picks


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = _summarize(root)
    out = root / "reports/normalization_summary.md"
    out.write_text(report + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
