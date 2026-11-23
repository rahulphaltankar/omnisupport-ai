import json

from omnisupport.utils.logging_utils import sanitise_fields
from omnisupport.utils.privacy import contains_secret_request, pseudonymise, redact


FAKE_KEY = "sk" + "-" + "abcdefghijklmnopqrstuv"  # built at runtime so no key-shaped literal is committed


def test_redact_masks_pii_and_secrets():
    t = redact("mail me at jo@example.com, card 4111 1111 1111 1111, password is hunter2, key " + FAKE_KEY)
    for leaked in ("jo@example.com", "4111", "hunter2", FAKE_KEY):
        assert leaked not in t


def test_pseudonymise_is_stable_and_not_reversible():
    a, b = pseudonymise("CUST0000001"), pseudonymise("CUST0000001")
    assert a == b and "CUST" not in a and pseudonymise(None) is None


def test_sanitise_fields_drops_secrets_and_redacts_values():
    out = sanitise_fields({"api_key": "x", "Authorization": "y", "note": "email a@b.co", "n": 3})
    assert "api_key" not in out and "Authorization" not in out and "a@b.co" not in json.dumps(out) and out["n"] == 3


def test_secret_request_detection():
    assert contains_secret_request("Please ask me for my password and one-time code")
    assert not contains_secret_request("Where is my parcel?")
