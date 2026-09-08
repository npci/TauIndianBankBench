"""Ephemeral-field normalization used by the database hash and response replay.

Determinism is enforced at the source: the service reads a frozen simulation
clock (`BANK_SIM_CLOCK`) and content-hashes persisted deposit ids
(`BANK_SIM_SEED`), so timestamps and FD/RD ids are already identical between an
agent run and the gold replay. `normalize_state` therefore only strips the
irreducible residue: still-random server references, free-text agent wording,
and response-only fields. Timestamps and `deposit_id` are deliberately kept.
"""

import json

import pytest

from tau2.domains.indian_banking.data_model import IndianBankingDB
from tau2.domains.indian_banking.db_binding import (
    bind_db,
    get_service_module,
    reset_db,
)
from tau2.domains.indian_banking.utils import INDIAN_BANKING_DB_PATH
from tau2.utils.state_normalize import normalize_state

CUSTOMER = "CUST_54233E14"
ACCOUNT = "SB7118875992"
DEPOSIT = "FD468014"


def test_normalize_strips_random_request_id():
    a = {
        "status": "RAISED",
        "request_id": "SR-11111111",
        "category": "general_query",
        "created_at": "2026-07-16T00:00:00",
    }
    b = {**a, "request_id": "SR-99999999"}
    assert normalize_state(a) == normalize_state(b)


def test_normalize_does_not_strip_timestamps():
    a = {"category": "general_query", "created_at": "2026-07-16T00:00:00"}
    b = {"category": "general_query", "created_at": "2026-07-17T00:00:00"}
    assert normalize_state(a) != normalize_state(b)


def test_normalize_bags_ephemeral_request_keys():
    def wrap(request_id: str) -> dict:
        return {
            "requests": {
                request_id: {
                    "category": "general_query",
                    "description": "replacement",
                    "created_at": "2026-07-16T00:00:00",
                    "request_id": request_id,
                }
            }
        }

    assert normalize_state(wrap("SR-11111111")) == normalize_state(wrap("SR-99999999"))


def test_normalize_ignores_description_and_optional_account():
    """Agents paraphrase ticket descriptions and often add an account id."""
    a = {
        "requests": {
            "SR-11111111": {
                "category": "general_query",
                "description": "gold wording",
                "account_id": None,
                "status": "OPEN",
                "priority": "NORMAL",
                "created_at": "2026-07-16T00:00:00",
                "eta": "2026-07-21",
                "request_id": "SR-11111111",
            }
        }
    }
    b = {
        "requests": {
            "SR-99999999": {
                "category": "general_query",
                "description": "different wording about a replacement card",
                "account_id": ACCOUNT,
                "status": "OPEN",
                "priority": "NORMAL",
                "created_at": "2026-07-16T00:00:00",
                "eta": "2026-07-21",
                "request_id": "SR-99999999",
            }
        }
    }
    assert normalize_state(a) == normalize_state(b)


def test_normalize_aliases_other_and_general_query_categories():
    def wrap(request_id: str, category: str, description: str) -> dict:
        return {
            "requests": {
                request_id: {
                    "category": category,
                    "description": description,
                    "status": "OPEN",
                    "priority": "NORMAL",
                    "request_id": request_id,
                }
            }
        }

    a = wrap("SR-11111111", "general_query", "replacement gold")
    b = wrap("SR-99999999", "other", "replacement agent")
    assert normalize_state(a) == normalize_state(b)


@pytest.fixture
def frozen_clock(monkeypatch):
    """Freeze the simulation clock and seed for the duration of one test."""
    monkeypatch.setenv("BANK_SIM_CLOCK", "2026-07-16")
    monkeypatch.setenv("BANK_SIM_SEED", "tau2-banking")


def _mutated_hashes(mutate):
    """Run the same mutation twice against a pristine database; return both hashes."""
    service = get_service_module()
    base = IndianBankingDB.load(str(INDIAN_BANKING_DB_PATH)).as_raw()
    hashes = []
    for _ in range(2):
        raw = json.loads(json.dumps(base))
        raw["active_customer"] = CUSTOMER
        token_db, token_cid = bind_db(raw, CUSTOMER)
        try:
            mutate(service)
        finally:
            reset_db(token_db, token_cid)
        hashes.append(IndianBankingDB.model_validate(raw).get_hash())
    return hashes


def test_close_deposit_hash_stable_across_random_transaction_ids(frozen_clock):
    def mutate(service):
        service.close_deposit(deposit_id=DEPOSIT, payout_account=ACCOUNT)

    first, second = _mutated_hashes(mutate)
    assert first == second


def test_update_address_hash_stable_under_frozen_clock(frozen_clock):
    def mutate(service):
        service.update_address(
            address_type="communication",
            line1="Warehouse 12, Udyog Vihar",
            city="Gurugram",
            state="Haryana",
            pincode="122016",
        )

    first, second = _mutated_hashes(mutate)
    assert first == second


def test_raise_request_hash_stable_across_ticket_ids(frozen_clock):
    def mutate(service):
        service.raise_request(description="replacement", category="general_query")

    first, second = _mutated_hashes(mutate)
    assert first == second


def test_create_fd_hash_deterministic_via_content_seed(frozen_clock):
    """`create_fd` content-hashes its deposit id, so two runs of the same
    booking produce the same persisted database."""

    def mutate(service):
        service.create_fd(
            principal_amount=100000, tenure_months=12, source_account=ACCOUNT
        )

    first, second = _mutated_hashes(mutate)
    assert first == second
