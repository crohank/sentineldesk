"""Builders for D4's tests (not collected by pytest)."""
from sentineldesk.safety.contracts import MaskedTicket, Precedent, Prediction, Triage

ANSWER_EN = (
    "Thank you for reaching out about the export problem. Could you please provide the exact error message "
    "and the version of the software you are using? We will then look into it."
)


def ticket(body: str, subject: str = "") -> MaskedTicket:
    return MaskedTicket(subject=subject, body=body)


def triage(type_="Request", tconf=0.9, queue="Technical Support", qconf=0.9, priority="medium", pconf=0.9,
           p_incident=None, p_high=None) -> Triage:
    tprobs = {type_: tconf}
    if p_incident is not None:
        tprobs["Incident"] = p_incident
    pprobs = {priority: pconf}
    if p_high is not None:
        pprobs["high"] = p_high
    return Triage(
        type=Prediction(type_, tconf, tprobs),
        queue=Prediction(queue, qconf, {queue: qconf}),
        priority=Prediction(priority, pconf, pprobs),
    )


def precedent(tid="TKT-01234", score=0.8, answer=ANSWER_EN, lang="en", subject="Export fails") -> Precedent:
    return Precedent(ticket_id=tid, score=score, subject=subject, answer_snippet=answer, answer_language=lang)
