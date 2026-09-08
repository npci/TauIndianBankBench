from typing import Any, Dict, Optional

from pydantic import Field

from tau2.environment.db import DB
from tau2.utils import get_dict_hash
from tau2.utils.state_normalize import normalize_state


class IndianBankingDB(DB):
    """Customer database for the Indian banking domain.

    Shape: ``{active_customer, customers: {id: {profile, login_context,
    accounts, ...}}}``. Nested customer records are kept as plain dicts so the
    tool implementations can mutate them in place.
    """

    active_customer: Optional[str] = Field(
        default=None, description="Logged-in customer id for this session"
    )
    customers: Dict[str, Any] = Field(
        default_factory=dict, description="Customer records keyed by customer id"
    )

    def as_raw(self) -> dict[str, Any]:
        """Mutable view used by the backend (customers dict is the same object)."""
        return {
            "active_customer": self.active_customer,
            "customers": self.customers,
        }

    def get_hash(self) -> str:
        """Hash the database after stripping server-generated ids and free text.

        Tools embed random ``reference_id`` / ``request_id`` keys and agent
        wording; a gold run and an agent run must still match when the semantic
        mutation is the same.
        """
        return get_dict_hash(normalize_state(self.model_dump()))

