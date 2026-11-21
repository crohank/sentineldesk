"""Write fixtures/draft_examples.json: real drafts from the release stack, for D4/D5.

    python scripts/make_draft_fixture.py

Uses the full-KB retriever (EmbeddingGemma, sparse fallback) and Generator.default()
(Gemini on Vertex when GOOGLE_CLOUD_PROJECT is set, then Gemma 4 via Ollama, then extractive).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentineldesk.generation import Generator  # noqa: E402
from sentineldesk.language import detect_language  # noqa: E402
from sentineldesk.retrieval import Retriever  # noqa: E402

MIN_TOP_SCORE = 0.35  # EmbeddingGemma relevance guard (D3-19)
CASES = {
    "routine_password_reset_en": {"subject": None, "body": "I forgot my password and can still access my registered email. How can I reset my password for the customer portal?"},
    "password_reset_de_from_english_evidence": {"subject": None, "body": "Ich habe mein Passwort für das Kundenportal vergessen und habe noch Zugriff auf meine registrierte E-Mail-Adresse. Wie kann ich das Passwort zurücksetzen?"},
    "printer_de": {"subject": None, "body": "Unser Drucker im Büro druckt seit gestern nur noch leere Seiten."},
    "invoice_en": {"subject": None, "body": "Where can I download last month's invoice for our subscription?"},
    "off_topic_abstains": {"subject": None, "body": "What is the weather in Paris tomorrow?"},
}


def main() -> None:
    r = Retriever.load(ROOT / "artifacts" / "retrieval" / "full", dense=ROOT / "artifacts" / "dense" / "embeddinggemma" / "full")
    gen = Generator.default()
    out = {"_note": "Real drafts from the release stack. Draft fields match D5's Draft schema; reply_text marks "
                    "citations inline as [TKT-#####]; every claim's evidence_span is a verbatim substring of that "
                    "precedent's answer_snippet. generator_mode / retrieval_mode say which backend served each case.",
           "generator_backends": [b.name for b in gen.backends], "min_top_score": MIN_TOP_SCORE, "cases": {}}
    for name, t in CASES.items():
        lang, _ = detect_language(f"{t['subject'] or ''}\n{t['body']}")
        precedents, rmode = r.retrieve_with_mode(t, 5)
        res = gen.draft_with_mode(t, lang, precedents, min_top_score=MIN_TOP_SCORE)
        out["cases"][name] = {"input": t, "language": lang, "retrieval_mode": rmode, "generator_mode": res.mode,
                              "generation_ms": res.ms, "draft": res.draft.to_dict(),
                              "precedent_ids": [p.ticket_id for p in precedents]}
        print(f"{name}: {res.mode} {res.ms:.0f} ms abstain={res.draft.abstain}")
    path = ROOT / "fixtures" / "draft_examples.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("->", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
