import pytest

from telco_churn.api.middleware import is_valid_request_id


@pytest.mark.parametrize(
    "value",
    ["abc123", "test-123", "550e8400-e29b-41d4-a716-446655440000", "a" * 64],
)
def test_valid_request_ids_accepted(value):
    assert is_valid_request_id(value)


@pytest.mark.parametrize(
    "value",
    [
        "",  # missing header
        "a" * 65,  # too long
        "has space",
        "line1\nFAKE LOG LINE",  # log injection attempt
        "semi;colon",
        "unicode-é",
    ],
)
def test_invalid_request_ids_rejected(value):
    assert not is_valid_request_id(value)