from sentineldesk.safety.contracts import TicketInput
from sentineldesk.safety.pii import mask_pii, mask_ticket


def masked(text: str) -> str:
    return mask_pii(text).text


def test_email():
    r = mask_pii("Contact me at jane.doe@example.com today")
    assert r.text == "Contact me at <email> today"
    assert r.counts == {"email": 1}


def test_phone_numbers():
    assert masked("Call +49 30 1234567 please") == "Call <tel_num> please"
    assert masked("My number is (555) 123-4567.") == "My number is <tel_num>."


def test_ip_addresses():
    assert masked("Server 192.168.10.25 is down") == "Server <ip_address> is down"
    assert masked("Host fe80::1ff:fe23:4567:890a fails") == "Host <ip_address> fails"


def test_iban_and_card():
    assert masked("IBAN DE89 3704 0044 0532 0130 00 please") == "IBAN <acc_num> please"
    assert masked("card 4111 1111 1111 1111 was charged") == "card <acc_num> was charged"
    # Fails the Luhn check and is too long for a phone number: left alone.
    assert masked("ref 1234 5678 9012 3456") == "ref 1234 5678 9012 3456"


def test_labelled_account_ids():
    assert masked("My customer number: AB12345") == "My customer number: <acc_num>"
    assert masked("Kundennummer 1234567, bitte prüfen") == "Kundennummer <acc_num>, bitte prüfen"
    assert masked("Check my account settings") == "Check my account settings"


def test_keeps_ticket_ids_error_codes_versions_dates():
    text = "TKT-01234 shows Error 0x80070005 and error 404 on version 10.2.3 since 2024-05-12."
    assert masked(text) == text


def test_empty_and_none():
    assert mask_pii(None).text == ""
    assert mask_pii("").counts == {}


def test_mask_ticket_handles_missing_subject():
    m = mask_ticket(TicketInput(body="mail me at a.b@example.de", subject=None))
    assert m.subject == ""
    assert m.body == "mail me at <email>"
