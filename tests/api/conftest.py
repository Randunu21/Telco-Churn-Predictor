import pytest

from telco_churn.api.schemas import EXAMPLE_CUSTOMER


@pytest.fixture
def valid_customer_payload():
    """A fresh dict per test, so a test that mutates it can't affect others."""
    return dict(EXAMPLE_CUSTOMER)