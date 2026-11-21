"""Write fixtures/precedents_example.json from the real full-KB index, for D4/D5.

    python scripts/make_precedent_fixture.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentineldesk.language import detect_language  # noqa: E402
from sentineldesk.retrieval import Retriever  # noqa: E402

CASES = {
    "routine_password_reset_en": {"subject": None, "body": "I forgot my password and can still access my registered email. How can I reset my password for the customer portal?"},
    "outage_vpn_hospital_de": {"subject": "VPN-Ausfall im Krankenhaus", "body": "Seit heute Morgen können sich alle Mitarbeitenden unserer Klinik nicht mehr per VPN verbinden. Telemedizin und der Zugriff auf die elektronische Patientenakte sind an allen Standorten ausgefallen."},
    "refund_demand_en": {"subject": None, "body": "Our service was interrupted. Confirm that we are entitled to a full refund and a guaranteed 25% SLA credit this month."},
    "password_reset_de": {"subject": None, "body": "Ich habe mein Passwort für das Kundenportal vergessen und habe noch Zugriff auf meine registrierte E-Mail-Adresse. Wie kann ich das Passwort zurücksetzen?"},
    "printer_de_crosslingual": {"subject": None, "body": "Unser Drucker im Büro druckt seit gestern nur noch leere Seiten."},
    "out_of_scope_en": {"subject": None, "body": "What is the weather in Paris tomorrow?"},
    "blank_body": {"subject": "Help", "body": "   "},
}


def main() -> None:
    r = Retriever.load(ROOT / "artifacts" / "retrieval" / "full",
                       dense=ROOT / "artifacts" / "dense" / "embeddinggemma" / "full",
                       vectordb=ROOT / "artifacts" / "qdrant" / "full")
    out = {
        "_note": "Real output of the full-KB release retriever (D3-12: EmbeddingGemma via Ollama, sparse fallback). "
                 "Fields follow PLAN.md Precedent. retrieval_mode per case says which mode served it. "
                 "Scores are cosine similarity and are only comparable within one retrieval mode. "
                 "answer_snippet still contains dataset mask tokens such as <tel_num>; D4 decides how to render them. "
                 "Full answers: Retriever.full_answer(ticket_id).",
        "release_mode": r.mode,
        "index_manifest": r.manifest,
        "cases": {},
    }
    for name, t in CASES.items():
        code, prob = detect_language(f"{t['subject'] or ''}\n{t['body']}")
        out["cases"][name] = {
            "input": t,
            "detected_language": {"code": code, "probability": prob},
            "precedents": [p.to_dict() for p in r.retrieve(t, 5)],
        }
        out["cases"][name]["retrieval_mode"] = r.last_mode
    path = ROOT / "fixtures" / "precedents_example.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("->", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
