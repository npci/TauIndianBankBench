"""Banking back office: the stateful implementation behind every tool.

Each public function here is one tool. They read and write the customer record
bound to the current call (see `db_binding.bind_db`) and return a JSON string.

Two environment variables make a run reproducible, and the domain sets both:
`BANK_SIM_CLOCK` freezes what the bank calls "today", and `BANK_SIM_SEED`
makes every server-assigned reference a function of the action rather than of
chance. Evaluation replays the same actions in a second process and compares
the resulting database, so neither may drift between runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from datetime import datetime, timedelta
from typing import Any, Literal, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
PERSIST_WRITES = os.environ.get("BANK_DB_PERSIST", "0") == "1"


def _customer_db_path() -> str:
    """Path of the customer database file, overridable with BANK_CUSTOMER_DB."""
    return os.environ.get("BANK_CUSTOMER_DB", os.path.join(_HERE, "db.json"))


def _bank_from_ifsc(ifsc: Any, fallback: Any = None) -> Optional[str]:
    """Name of the bank an IFSC belongs to, or the caller-supplied fallback."""
    if fallback:
        return str(fallback)
    code = str(ifsc or "").strip().upper()[:4]
    if not code:
        return None
    return f"Unknown Bank ({code})"

# ---------------------------------------------------------------------------
# Bank-level reference data (NOT customer data). Authoritative rate/product
# sources used by the calculators and rate tools.
# ---------------------------------------------------------------------------

DEPOSIT_PENALTY_RATE = 1.0  # % deducted from the contract rate on premature closure
GOLD_LTV_CAP = 75.0  # RBI cap, applied internally by calculate_gold_loan_ltv

# Rate cards keyed by product_type. Deposit slabs are by tenure (months); loan
# slabs are by amount. "senior_rate" only applies to deposits.
RATE_CARDS: dict[str, dict[str, Any]] = {
    "fd": {
        "unit": "tenure_months",
        "compounding": "quarterly",
        "slabs": [
            {"label": "7 days - 3 months", "min_months": 0, "max_months": 3, "general_rate": 4.5, "senior_rate": 5.0},
            {"label": "3 - 6 months", "min_months": 3, "max_months": 6, "general_rate": 5.5, "senior_rate": 6.0},
            {"label": "6 - 12 months", "min_months": 6, "max_months": 12, "general_rate": 6.8, "senior_rate": 7.3},
            {"label": "12 - 24 months", "min_months": 12, "max_months": 24, "general_rate": 7.1, "senior_rate": 7.6},
            {"label": "24 - 36 months", "min_months": 24, "max_months": 36, "general_rate": 7.3, "senior_rate": 7.8},
            {"label": "36 - 60 months", "min_months": 36, "max_months": 60, "general_rate": 7.0, "senior_rate": 7.5},
        ],
    },
    "rd": {
        "unit": "tenure_months",
        "compounding": "quarterly",
        "slabs": [
            {"label": "6 - 12 months", "min_months": 6, "max_months": 12, "general_rate": 6.5, "senior_rate": 7.0},
            {"label": "12 - 24 months", "min_months": 12, "max_months": 24, "general_rate": 6.8, "senior_rate": 7.3},
            {"label": "24 - 36 months", "min_months": 24, "max_months": 36, "general_rate": 7.0, "senior_rate": 7.5},
            {"label": "36 - 60 months", "min_months": 36, "max_months": 60, "general_rate": 6.8, "senior_rate": 7.3},
        ],
    },
    "home_loan": {
        "unit": "amount",
        "slabs": [
            {"label": "up to 30 lakh", "min_amount": 0, "max_amount": 3000000, "general_rate": 8.5},
            {"label": "30 - 75 lakh", "min_amount": 3000000, "max_amount": 7500000, "general_rate": 8.35},
            {"label": "above 75 lakh", "min_amount": 7500000, "max_amount": 1_000_000_000, "general_rate": 8.6},
        ],
    },
    "personal_loan": {
        "unit": "amount",
        "slabs": [
            {"label": "up to 5 lakh", "min_amount": 0, "max_amount": 500000, "general_rate": 12.5},
            {"label": "5 - 15 lakh", "min_amount": 500000, "max_amount": 1500000, "general_rate": 11.5},
            {"label": "above 15 lakh", "min_amount": 1500000, "max_amount": 1_000_000_000, "general_rate": 10.75},
        ],
    },
    "gold_loan": {
        "unit": "amount",
        "slabs": [
            {"label": "up to 3 lakh", "min_amount": 0, "max_amount": 300000, "general_rate": 8.5},
            {"label": "3 - 10 lakh", "min_amount": 300000, "max_amount": 1000000, "general_rate": 8.0},
            {"label": "above 10 lakh", "min_amount": 1000000, "max_amount": 1_000_000_000, "general_rate": 7.5},
        ],
    },
}

GOLD_RATES_PER_GRAM = {18: 4600, 22: 5850, 24: 6300}

# Bank product catalog (customer-agnostic). Personalised offers come from the
# customer record's "offers" list.
PRODUCT_CATALOG = [
    {"type": "deposit", "code": "FD", "name": "Fixed Deposit",
     "summary": "Lump-sum deposit with guaranteed returns; flexible tenure 7 days to 10 years."},
    {"type": "deposit", "code": "RD", "name": "Recurring Deposit",
     "summary": "Systematic monthly savings from ₹500/month; interest paid at maturity."},
    {"type": "loan", "code": "HL", "name": "Home Loan",
     "summary": "Financing for purchase/construction; floating rates linked to external benchmark."},
    {"type": "loan", "code": "PL", "name": "Personal Loan",
     "summary": "Unsecured loan for any need; quick disbursal, minimal documentation."},
    {"type": "loan", "code": "GL", "name": "Gold Loan",
     "summary": "Loan against gold ornaments up to the RBI LTV cap; fast processing."},
    {"type": "card", "code": "DC", "name": "Debit Card",
     "summary": "Linked to your account with configurable per-channel limits."},
    {"type": "card", "code": "CC", "name": "Credit Card",
     "summary": "Revolving credit with rewards and EMI conversion options."},
    {"type": "insurance", "code": "LI", "name": "Life Insurance (Bancassurance)",
     "summary": "Term and endowment cover distributed through the bank."},
]


# ---------------------------------------------------------------------------
# Data-access layer (customer DB + KB). Kept deliberately thin and pluggable.
# ---------------------------------------------------------------------------

_DB_CACHE: dict[str, Any] | None = None


def _load_db() -> dict[str, Any]:
    """Load and cache the customer database. Empty DB if the file is absent."""
    global _DB_CACHE
    if _DB_CACHE is None:
        path = _customer_db_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                _DB_CACHE = json.load(fh)
        else:
            _DB_CACHE = {"customers": {}}
    return _DB_CACHE


def _persist_db() -> None:
    if PERSIST_WRITES and _DB_CACHE is not None:
        with open(_customer_db_path(), "w", encoding="utf-8") as fh:
            json.dump(_DB_CACHE, fh, indent=2)


def _active_customer_id() -> Optional[str]:
    db = _load_db()
    customers = db.get("customers", {})
    if not customers:
        return None
    cid = os.environ.get("BANK_ACTIVE_CUSTOMER") or db.get("active_customer")
    if cid and cid in customers:
        return cid
    if len(customers) == 1:
        return next(iter(customers))
    return cid  # may be None / unknown → resolved as CUSTOMER_NOT_FOUND downstream


def _active_customer() -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Return (customer_record, error_json). Exactly one is non-None."""
    db = _load_db()
    if not db.get("customers"):
        return None, _error("NO_CUSTOMER_DB",
                            "No customer database configured. Set BANK_CUSTOMER_DB to a customer DB JSON file.")
    cid = _active_customer_id()
    cust = db.get("customers", {}).get(cid) if cid else None
    if not cust:
        return None, _error("CUSTOMER_NOT_FOUND",
                            "No active customer resolved. Set BANK_ACTIVE_CUSTOMER to a valid customer id.")
    return cust, None


def _login_context(cust: dict[str, Any]) -> dict[str, Any]:
    ctx = cust.get("login_context", {})
    return {
        "linked_accounts": ctx.get("linked_accounts", list(cust.get("accounts", {}).keys())),
        "linked_products": ctx.get("linked_products",
                                   list(cust.get("deposits", {}).keys())
                                   + list(cust.get("loans", {}).keys())
                                   + list(cust.get("insurance", {}).keys())),
        "linked_cards": ctx.get("linked_cards", list(cust.get("cards", {}).keys())),
    }


def _error(code: str, message: str, retriable: bool = False) -> str:
    return json.dumps({"error_code": code, "error_message": message, "retriable": retriable})


def _now() -> datetime:
    """Wall clock, or a frozen simulation clock when ``BANK_SIM_CLOCK`` is set.

    Evaluation harnesses build a "gold" DB by replaying expected actions in a
    separate process seconds after the agent run; wall-clock timestamps would
    then differ and break the DB-equality check even for identical behavior.
    When ``BANK_SIM_CLOCK`` (an ISO datetime, e.g. ``2026-07-16`` or
    ``2026-07-16T09:00:00``) is present, return that fixed instant so the
    environment is a deterministic function of (initial state, actions).
    Unset (production serving) -> real wall clock, unchanged.
    """
    sim = os.environ.get("BANK_SIM_CLOCK")
    if sim:
        # Fail loud on a malformed value rather than silently reverting to the
        # wall clock, which would reintroduce non-deterministic timestamps.
        return datetime.fromisoformat(sim)
    return datetime.now()


def _today_str() -> str:
    return _now().strftime("%Y-%m-%d")


def _ref(prefix: str, digits: int = 8, key: Optional[str] = None) -> str:
    """Server-assigned reference id.

    Normally random. When ``BANK_SIM_SEED`` is set AND a semantic ``key`` is
    provided, derive the id deterministically from ``seed + prefix + key`` so
    that the same semantic mutation yields the same id in both the agent run
    and the gold replay (which run in separate processes and cannot share an
    RNG stream). Ids that only appear as bag-normalized dict keys or as
    stripped response fields need no key; only ids persisted into ordered
    structures (e.g. ``login_context.linked_products``) require one.
    Unset seed, or no key -> random, unchanged.
    """
    seed = os.environ.get("BANK_SIM_SEED")
    if seed and key is not None:
        digest = hashlib.sha1(f"{seed}:{prefix}:{key}".encode("utf-8")).hexdigest()
        span = 10 ** digits - 10 ** (digits - 1)
        return f"{prefix}{10 ** (digits - 1) + int(digest, 16) % span}"
    return f"{prefix}{random.randint(10 ** (digits - 1), 10 ** digits - 1)}"


# ---------------------------------------------------------------------------
# 1. Knowledge base
# ---------------------------------------------------------------------------

# The corpus is a directory of markdown documents, one per topic, each split
# into `##` sections. Files are read in name order and sections in document
# order, so ranking ties always break the same way.
KB_DIR = os.environ.get("BANK_KB_DIR", "")

_KB_CACHE: Optional[list[dict[str, Any]]] = None


def _kb_dir() -> str:
    if KB_DIR:
        return KB_DIR
    from tau2.domains.indian_banking.utils import INDIAN_BANKING_KB_DIR

    return str(INDIAN_BANKING_KB_DIR)


def _load_kb() -> list[dict[str, Any]]:
    """Load `kb/*.md` as one searchable section per `##` heading."""
    global _KB_CACHE
    if _KB_CACHE is not None:
        return _KB_CACHE
    from pathlib import Path

    directory = Path(_kb_dir())
    provenance: dict[str, dict[str, Any]] = {}
    prov_file = directory / "provenance.json"
    if prov_file.is_file():
        for entry in json.loads(prov_file.read_text(encoding="utf-8")):
            provenance[str(entry.get("topic", ""))] = entry

    articles: list[dict[str, Any]] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            match = re.match(r"#\s+(.+)", text)
            doc_title = (
                match.group(1).strip() if match else path.stem.replace("_", " ").title()
            )
            entry = provenance.get(path.stem, {})
            urls = entry.get("urls") or []
            source_url = urls[0] if urls else f"kb/{path.name}"
            source_date = str(entry.get("retrieved", ""))
            for section in re.split(r"\n##\s+", text)[1:]:
                header, _, body = section.partition("\n")
                body = body.strip()
                if not body:
                    continue
                articles.append(
                    {
                        "text": body[:1200],
                        "source_url": source_url,
                        "source_date": source_date,
                        "bank_id": "",
                        "category": doc_title,
                        "label": header.strip(),
                        "relevance": "YES",
                    }
                )
    _KB_CACHE = articles
    return _KB_CACHE


def _kb_score(article: dict[str, Any], tokens: list[str]) -> float:
    """Fraction of query tokens present in the section, mapped to 0.5 - 0.9."""
    text = f"{article['label']} {article['text']}".lower()
    hits = sum(1 for token in tokens if token in text)
    coverage = min(1.0, hits / max(1, len(tokens)))
    return round(0.5 + 0.4 * coverage, 4)


def search_knowledge_base(
    query: str,
    bank_id: Optional[str] = None,
    category: Optional[str] = None,
    top_k: Optional[int] = None,
) -> str:
    """Search the bank's knowledge corpus (RBI circulars, product docs, scheme
    details, FAQs, how-to guides) with a natural-language query, backed by a
    semantic RAG index (bge-m3 embeddings + Qdrant + cross-encoder reranker).

    Only ``query`` is required. Optionally narrow with ``bank_id`` (a datapoint id
    like "BNK-0001"), ``category`` (a search type like "Regulatory"), or ``top_k``
    (number of results). Returns the most relevant chunks, each with its
    ``source_url`` and a relevance ``score``."""
    articles = _load_kb()
    if not articles:
        return json.dumps(
            {
                "query": query,
                "total_results": 0,
                "results": [],
                "note": "Knowledge base is not available.",
            }
        )

    pool = articles
    if category:
        wanted = category.strip().lower()
        narrowed = [a for a in pool if wanted in a["category"].lower()]
        if narrowed:
            pool = narrowed
    if bank_id:
        wanted = bank_id.strip().lower()
        narrowed = [a for a in pool if wanted == a["bank_id"].lower()]
        if narrowed:
            pool = narrowed

    tokens = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t]
    limit = top_k or 3
    scored = sorted(pool, key=lambda a: _kb_score(a, tokens), reverse=True)
    matched = [a for a in scored if _kb_score(a, tokens) > 0.5][:limit]
    if not matched:
        matched = scored[: min(3, limit)]

    results = []
    for article in matched:
        score = _kb_score(article, tokens)
        results.append({**article, "score": score, "similarity": score})
    return json.dumps({"query": query, "total_results": len(results), "results": results})


# ---------------------------------------------------------------------------
# 2-4. Accounts
# ---------------------------------------------------------------------------

def get_account_balance(account_ids: list[str]) -> str:
    """Return current available balance and hold/lien amount for one or more
    linked accounts."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    accounts = cust.get("accounts", {})
    out, errors = [], []
    for aid in account_ids:
        if aid not in ctx["linked_accounts"] or aid not in accounts:
            errors.append({"account_id": aid, "error": "Account not linked to logged-in customer"})
            continue
        acc = accounts[aid]
        out.append({
            "account_id": aid,
            "available_balance": acc.get("available_balance", 0.0),
            "hold_amount": acc.get("hold_amount", 0.0),
            "currency": acc.get("currency", "INR"),
            "as_of": _now().isoformat(),
        })
    return json.dumps({"balances": out, "errors": errors})


def get_account_details(account_ids: list[str]) -> str:
    """Return account metadata — bank name, type, status, home branch, IFSC,
    holders, nominee, open date. Does not return balances or transactions."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    accounts = cust.get("accounts", {})
    out, errors = [], []
    for aid in account_ids:
        if aid not in ctx["linked_accounts"] or aid not in accounts:
            errors.append({"account_id": aid, "error": "Account not linked to logged-in customer"})
            continue
        acc = accounts[aid]
        out.append({
            "account_id": aid,
            "account_type": acc.get("account_type"),
            "status": acc.get("status", "ACTIVE"),
            "bank_name": acc.get("bank_name") or _bank_from_ifsc(acc.get("ifsc")),
            "ifsc": acc.get("ifsc"),
            "home_branch": acc.get("home_branch"),
            "holders": acc.get("holders", []),
            "nominee": acc.get("nominee"),
            "open_date": acc.get("open_date"),
            "currency": acc.get("currency", "INR"),
        })
    return json.dumps({"accounts": out, "errors": errors})


def get_transaction_history(
    account_id: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    type: Literal["debit", "credit", "all"] = "all",
    channel: Literal["upi", "neft", "imps", "rtgs", "atm", "pos", "cheque", "all"] = "all",
    limit: int = 10,
    offset: int = 0,
) -> str:
    """Return statement entries for a single account, newest first, with
    date/direction/channel filters and pagination."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    if account_id not in ctx["linked_accounts"] or account_id not in cust.get("accounts", {}):
        return _error("VALIDATION_ERROR", f"Account {account_id} not linked to logged-in customer")
    txns = list(cust["accounts"][account_id].get("transactions", []))
    if from_date:
        txns = [t for t in txns if t.get("date", "") >= from_date]
    if to_date:
        txns = [t for t in txns if t.get("date", "") <= to_date]
    if type != "all":
        txns = [t for t in txns if t.get("direction") == type]
    if channel != "all":
        txns = [t for t in txns if t.get("channel") == channel]
    txns.sort(key=lambda t: t.get("date", ""), reverse=True)
    total = len(txns)
    page = txns[offset: offset + limit]
    return json.dumps({
        "account_id": account_id,
        "total_records": total,
        "returned": len(page),
        "offset": offset,
        "limit": limit,
        "transactions": page,
    })


# ---------------------------------------------------------------------------
# 5-6. Deposit read
# ---------------------------------------------------------------------------

def _get_deposits(cust: dict[str, Any], deposit_ids: list[str], kind: Optional[str]) -> tuple[list, list]:
    ctx = _login_context(cust)
    deposits = cust.get("deposits", {})
    out, errors = [], []
    for did in deposit_ids:
        dep = deposits.get(did)
        if not dep or did not in ctx["linked_products"]:
            errors.append({"deposit_id": did, "error": "Deposit not linked to logged-in customer"})
            continue
        if kind and dep.get("kind") != kind:
            errors.append({"deposit_id": did, "error": f"{did} is not a {kind}"})
            continue
        out.append(dep)
    return out, errors


def get_fd_details(deposit_ids: list[str]) -> str:
    """Return full details of the customer's Fixed Deposits."""
    cust, err = _active_customer()
    if err:
        return err
    fds, errors = _get_deposits(cust, deposit_ids, "FD")
    return json.dumps({"fds": fds, "errors": errors})


def get_rd_details(deposit_ids: list[str]) -> str:
    """Return full details of the customer's Recurring Deposits, including
    installment amount, installments paid, and total accumulated to date."""
    cust, err = _active_customer()
    if err:
        return err
    rds, errors = _get_deposits(cust, deposit_ids, "RD")
    return json.dumps({"rds": rds, "errors": errors})


# ---------------------------------------------------------------------------
# 7. Rate card
# ---------------------------------------------------------------------------

def get_deposit_loan_rates(
    product_type: Literal["fd", "rd", "home_loan", "personal_loan", "gold_loan"],
    tenure_months: Optional[int] = None,
    amount: Optional[float] = None,
) -> str:
    """Return the current rate card for a deposit or loan product, by
    tenure/amount slab. The sole authoritative source for any rate used in a
    calculator or a booking."""
    card = RATE_CARDS.get(product_type)
    if not card:
        return _error("VALIDATION_ERROR", f"No rate card for product {product_type}")
    slabs = card["slabs"]
    if card["unit"] == "tenure_months" and tenure_months is not None:
        slabs = [s for s in slabs if s["min_months"] < tenure_months <= s["max_months"]] or slabs
    if card["unit"] == "amount" and amount is not None:
        slabs = [s for s in slabs if s["min_amount"] <= amount < s["max_amount"]] or slabs
    return json.dumps({
        "product_type": product_type,
        "effective_date": _today_str(),
        "unit": card["unit"],
        "compounding": card.get("compounding"),
        "slabs": slabs,
    })


# ---------------------------------------------------------------------------
# 8-9. Deposit calculators (pure)
# ---------------------------------------------------------------------------

def calculate_fd_maturity(
    principal_amount: float,
    rate: float,
    tenure_months: int,
    payout_type: Literal["cumulative", "monthly", "quarterly"] = "cumulative",
) -> str:
    """Compute maturity value and interest earned for a Fixed Deposit.
    Compounding is quarterly (bank standard)."""
    years = tenure_months / 12.0
    if payout_type == "cumulative":
        n = 4  # quarterly compounding
        maturity = principal_amount * (1 + rate / 100 / n) ** (n * years)
        total_interest = maturity - principal_amount
        return json.dumps({
            "principal_amount": principal_amount,
            "rate": rate,
            "tenure_months": tenure_months,
            "payout_type": payout_type,
            "compounding": "quarterly",
            "maturity_amount": round(maturity, 2),
            "total_interest": round(total_interest, 2),
        })
    # Non-cumulative: simple interest paid out periodically, principal returned at maturity.
    periods_per_year = 12 if payout_type == "monthly" else 4
    total_interest = principal_amount * rate / 100 * years
    payout_each = principal_amount * rate / 100 / periods_per_year
    return json.dumps({
        "principal_amount": principal_amount,
        "rate": rate,
        "tenure_months": tenure_months,
        "payout_type": payout_type,
        "periodic_payout": round(payout_each, 2),
        "num_payouts": int(round(periods_per_year * years)),
        "total_interest": round(total_interest, 2),
        "maturity_amount": round(principal_amount, 2),
    })


def calculate_rd_maturity(monthly_installment: float, rate: float, tenure_months: int) -> str:
    """Compute maturity value and interest earned for a Recurring Deposit.
    Interest is paid at maturity; compounding is quarterly."""
    quarterly_rate = rate / 100 / 4
    balance = 0.0
    for month in range(1, tenure_months + 1):
        balance += monthly_installment
        if month % 3 == 0:
            balance += balance * quarterly_rate
    # Interest for any trailing partial quarter.
    trailing = tenure_months % 3
    if trailing:
        balance += balance * quarterly_rate * (trailing / 3)
    total_deposited = monthly_installment * tenure_months
    return json.dumps({
        "monthly_installment": monthly_installment,
        "rate": rate,
        "tenure_months": tenure_months,
        "compounding": "quarterly",
        "total_deposited": round(total_deposited, 2),
        "maturity_amount": round(balance, 2),
        "total_interest": round(balance - total_deposited, 2),
    })


# ---------------------------------------------------------------------------
# 10-11. Deposit booking (mutation)
# ---------------------------------------------------------------------------

def _debit_account(cust: dict[str, Any], account_id: str, amount: float, description: str,
                   channel: str = "neft") -> Optional[str]:
    """Debit an account in the in-memory store and append a transaction.
    Returns an error string on failure, else None."""
    acc = cust.get("accounts", {}).get(account_id)
    if not acc:
        return _error("VALIDATION_ERROR", f"Source account {account_id} not found")
    balance = acc.get("available_balance", 0.0)
    if balance < amount:
        return _error("INSUFFICIENT_FUNDS",
                      f"Available balance {balance} is less than required {amount}")
    acc["available_balance"] = round(balance - amount, 2)
    acc.setdefault("transactions", []).append({
        "reference_id": _ref("TXN", 10),
        "date": _today_str(),
        "description": description,
        "amount": -abs(amount),
        "direction": "debit",
        "channel": channel,
        "balance_after": acc["available_balance"],
    })
    return None


def create_fd(
    principal_amount: float,
    tenure_months: int,
    source_account: str,
    payout_type: Literal["cumulative", "monthly", "quarterly"] = "cumulative",
    maturity_instruction: Literal["no_renewal", "renew_principal", "renew_principal_interest"] = "no_renewal",
    nominee: Optional[dict[str, Any]] = None,
) -> str:
    """Book a new Fixed Deposit, debiting principal_amount from the source
    account as a one-time transaction."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    if source_account not in ctx["linked_accounts"]:
        return _error("VALIDATION_ERROR", f"Source account {source_account} not linked to customer")
    senior = bool(cust.get("profile", {}).get("senior_citizen"))
    slabs = [s for s in RATE_CARDS["fd"]["slabs"] if s["min_months"] < tenure_months <= s["max_months"]]
    rate = (slabs[0]["senior_rate"] if senior else slabs[0]["general_rate"]) if slabs else 7.0
    debit_err = _debit_account(cust, source_account, principal_amount, "FD booking", "internal")
    if debit_err:
        return debit_err
    maturity = principal_amount * (1 + rate / 100 / 4) ** (4 * tenure_months / 12)
    start = _now()
    maturity_date = (start + timedelta(days=int(tenure_months * 30.4375)))
    fd_id = _ref(
        "FD", 6,
        key=f"{cust.get('customer_id')}|{principal_amount}|{tenure_months}|{source_account}",
    )
    record = {
        "deposit_id": fd_id, "kind": "FD", "principal": principal_amount, "rate": rate,
        "tenure_months": tenure_months, "start_date": start.strftime("%Y-%m-%d"),
        "maturity_date": maturity_date.strftime("%Y-%m-%d"), "maturity_amount": round(maturity, 2),
        "payout_type": payout_type, "maturity_instruction": maturity_instruction,
        "status": "ACTIVE", "payout_account": source_account, "nominee": nominee,
    }
    cust.setdefault("deposits", {})[fd_id] = record
    cust.setdefault("login_context", {}).setdefault("linked_products", []).append(fd_id)
    _persist_db()
    return json.dumps({
        "status": "BOOKED", "deposit_id": fd_id, "principal_amount": principal_amount,
        "rate": rate, "tenure_months": tenure_months, "payout_type": payout_type,
        "maturity_date": record["maturity_date"], "maturity_amount": round(maturity, 2),
        "source_account": source_account, "maturity_instruction": maturity_instruction,
        "receipt_ref": _ref("RCPT", 8), "booked_at": _now().isoformat(),
    })


def create_rd(
    monthly_installment: float,
    tenure_months: int,
    source_account: str,
    maturity_instruction: Literal["no_renewal", "renew_principal", "renew_principal_interest"] = "no_renewal",
    nominee: Optional[dict[str, Any]] = None,
) -> str:
    """Book a new Recurring Deposit. Authorizes a recurring monthly debit for the
    full tenure; the first installment is debited at booking."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    if source_account not in ctx["linked_accounts"]:
        return _error("VALIDATION_ERROR", f"Source account {source_account} not linked to customer")
    senior = bool(cust.get("profile", {}).get("senior_citizen"))
    slabs = [s for s in RATE_CARDS["rd"]["slabs"] if s["min_months"] < tenure_months <= s["max_months"]]
    rate = (slabs[0]["senior_rate"] if senior else slabs[0]["general_rate"]) if slabs else 6.8
    debit_err = _debit_account(cust, source_account, monthly_installment, "RD first installment", "internal")
    if debit_err:
        return debit_err
    # Reuse the maturity math.
    quarterly_rate = rate / 100 / 4
    balance = 0.0
    for month in range(1, tenure_months + 1):
        balance += monthly_installment
        if month % 3 == 0:
            balance += balance * quarterly_rate
    start = _now()
    maturity_date = (start + timedelta(days=int(tenure_months * 30.4375)))
    rd_id = _ref(
        "RD", 6,
        key=f"{cust.get('customer_id')}|{monthly_installment}|{tenure_months}|{source_account}",
    )
    record = {
        "deposit_id": rd_id, "kind": "RD", "monthly_installment": monthly_installment, "rate": rate,
        "tenure_months": tenure_months, "installments_paid": 1,
        "total_deposited": monthly_installment, "start_date": start.strftime("%Y-%m-%d"),
        "maturity_date": maturity_date.strftime("%Y-%m-%d"), "maturity_amount": round(balance, 2),
        "maturity_instruction": maturity_instruction, "status": "ACTIVE",
        "source_account": source_account, "payout_account": source_account, "nominee": nominee,
    }
    cust.setdefault("deposits", {})[rd_id] = record
    cust.setdefault("login_context", {}).setdefault("linked_products", []).append(rd_id)
    _persist_db()
    return json.dumps({
        "status": "BOOKED", "deposit_id": rd_id, "monthly_installment": monthly_installment,
        "rate": rate, "tenure_months": tenure_months, "first_installment_debited": monthly_installment,
        "recurring_commitment": f"₹{monthly_installment:.0f}/month for {tenure_months} months",
        "maturity_date": record["maturity_date"], "projected_maturity_amount": round(balance, 2),
        "source_account": source_account, "maturity_instruction": maturity_instruction,
        "receipt_ref": _ref("RCPT", 8), "booked_at": _now().isoformat(),
    })


# ---------------------------------------------------------------------------
# 12-14. Deposit closure / renewal
# ---------------------------------------------------------------------------

def _closure_quote_for(dep: dict[str, Any]) -> dict[str, Any]:
    today = _now()
    start = datetime.strptime(dep["start_date"], "%Y-%m-%d")
    maturity_date = datetime.strptime(dep["maturity_date"], "%Y-%m-%d")
    matured = today >= maturity_date
    if dep.get("kind") == "RD":
        principal = dep.get("total_deposited", dep.get("monthly_installment", 0) * dep.get("installments_paid", 0))
    else:
        principal = dep.get("principal", 0)
    contract_rate = dep.get("rate", 0.0)
    if matured:
        penalty_rate = 0.0
        payout = dep.get("maturity_amount", principal)
        interest = payout - principal
        penalty_amount = 0.0
    else:
        penalty_rate = DEPOSIT_PENALTY_RATE
        effective_rate = max(contract_rate - penalty_rate, 0.0)
        days_held = max((today - start).days, 0)
        interest = principal * effective_rate / 100 * days_held / 365
        penalty_amount = 0.0  # penalty is expressed as the rate reduction
        payout = principal + interest
    return {
        "deposit_id": dep["deposit_id"],
        "kind": dep.get("kind"),
        "matured": matured,
        "principal": round(principal, 2),
        "contract_rate": contract_rate,
        "penalty_rate": penalty_rate,
        "effective_rate": round(max(contract_rate - penalty_rate, 0.0), 2) if not matured else contract_rate,
        "interest_payable": round(interest, 2),
        "penalty_amount": round(penalty_amount, 2),
        "net_payout": round(payout, 2),
        "quote_date": _today_str(),
    }


def get_deposit_closure_quote(deposit_ids: list[str]) -> str:
    """Quote the payout if an FD or RD were closed today. Before maturity a
    penalty applies (rate recalculated at the penalised effective rate); after
    maturity no penalty applies. Read-only."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    deposits = cust.get("deposits", {})
    quotes, errors = [], []
    for did in deposit_ids:
        dep = deposits.get(did)
        if not dep or did not in ctx["linked_products"]:
            errors.append({"deposit_id": did, "error": "Deposit not linked to logged-in customer"})
            continue
        quotes.append(_closure_quote_for(dep))
    return json.dumps({"quotes": quotes, "errors": errors})


def close_deposit(deposit_id: str, payout_account: Optional[str] = None) -> str:
    """Close an FD or RD and credit proceeds to the payout account. Works for
    premature (penalty) and post-maturity (no penalty) closure. For an RD this
    also permanently stops all future installment debits."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    dep = cust.get("deposits", {}).get(deposit_id)
    if not dep or deposit_id not in ctx["linked_products"]:
        return _error("VALIDATION_ERROR", f"Deposit {deposit_id} not linked to logged-in customer")
    if dep.get("status") == "CLOSED":
        return _error("ALREADY_CLOSED", f"Deposit {deposit_id} is already closed")
    quote = _closure_quote_for(dep)
    credit_account = payout_account or dep.get("payout_account")
    if credit_account and credit_account in cust.get("accounts", {}):
        acc = cust["accounts"][credit_account]
        acc["available_balance"] = round(acc.get("available_balance", 0.0) + quote["net_payout"], 2)
        acc.setdefault("transactions", []).append({
            "reference_id": _ref("TXN", 10), "date": _today_str(),
            "description": f"{dep.get('kind')} {deposit_id} closure proceeds",
            "amount": quote["net_payout"], "direction": "credit", "channel": "internal",
            "balance_after": acc["available_balance"],
        })
    dep["status"] = "CLOSED"
    dep["closed_on"] = _today_str()
    _persist_db()
    return json.dumps({
        "status": "CLOSED", "deposit_id": deposit_id, "kind": dep.get("kind"),
        "premature": not quote["matured"], "penalty_rate": quote["penalty_rate"],
        "net_payout": quote["net_payout"], "payout_account": credit_account,
        "future_installments_stopped": dep.get("kind") == "RD",
        "closed_at": _now().isoformat(),
    })


def update_deposit_renewal(
    deposit_id: str,
    instruction: Literal["no_renewal", "renew_principal", "renew_principal_interest"],
) -> str:
    """Set the maturity instruction on an FD or RD. Takes effect on the maturity
    date, not immediately."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    dep = cust.get("deposits", {}).get(deposit_id)
    if not dep or deposit_id not in ctx["linked_products"]:
        return _error("VALIDATION_ERROR", f"Deposit {deposit_id} not linked to logged-in customer")
    previous = dep.get("maturity_instruction")
    dep["maturity_instruction"] = instruction
    _persist_db()
    return json.dumps({
        "status": "UPDATED", "deposit_id": deposit_id,
        "previous_instruction": previous, "new_instruction": instruction,
        "effective_on": dep.get("maturity_date"), "updated_at": _now().isoformat(),
    })


# ---------------------------------------------------------------------------
# 15-17. Loans
# ---------------------------------------------------------------------------

def get_loan_details(loan_ids: list[str]) -> str:
    """Return current loan state: outstanding balance, EMI, rate, remaining
    tenure, next due date."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    loans = cust.get("loans", {})
    out, errors = [], []
    for lid in loan_ids:
        loan = loans.get(lid)
        if not loan or lid not in ctx["linked_products"]:
            errors.append({"loan_id": lid, "error": "Loan not linked to logged-in customer"})
            continue
        out.append(loan)
    return json.dumps({"loans": out, "errors": errors})


def get_loan_foreclosure_quote(loan_ids: list[str], as_of_date: Optional[str] = None) -> str:
    """Quote the total payoff to close a loan early as of a given date.
    Informational only. Also use this when a customer asks to 'pause', 'stop', or
    'reduce' EMIs, since no such capability exists."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    loans = cust.get("loans", {})
    value_date = as_of_date or _today_str()
    quotes, errors = [], []
    for lid in loan_ids:
        loan = loans.get(lid)
        if not loan or lid not in ctx["linked_products"]:
            errors.append({"loan_id": lid, "error": "Loan not linked to logged-in customer"})
            continue
        outstanding = loan.get("outstanding", 0.0)
        rate = loan.get("rate", 0.0)
        accrued_interest = outstanding * rate / 100 * 30 / 365
        # Foreclosure charge: nil for floating home loans, else ~2%.
        charge_rate = 0.0 if loan.get("loan_type") == "home_loan" else 2.0
        foreclosure_charges = outstanding * charge_rate / 100
        gst = foreclosure_charges * 0.18
        total = outstanding + accrued_interest + foreclosure_charges + gst
        quotes.append({
            "loan_id": lid, "as_of_date": value_date, "loan_type": loan.get("loan_type"),
            "outstanding_principal": round(outstanding, 2),
            "accrued_interest": round(accrued_interest, 2),
            "foreclosure_charges": round(foreclosure_charges, 2),
            "gst_on_charges": round(gst, 2),
            "total_payable": round(total, 2),
        })
    return json.dumps({"quotes": quotes, "errors": errors,
                       "note": "No tool closes a loan; route the actual foreclosure via raise_request."})


def calculate_emi(principal: float, rate: float, tenure_months: int) -> str:
    """Compute the fixed monthly EMI, total interest, total repayment, and full
    amortisation schedule for a loan."""
    r = rate / 100 / 12
    if r == 0:
        emi = principal / tenure_months
    else:
        emi = principal * r * (1 + r) ** tenure_months / ((1 + r) ** tenure_months - 1)
    schedule = []
    balance = principal
    for month in range(1, tenure_months + 1):
        interest_component = balance * r
        principal_component = emi - interest_component
        balance = balance - principal_component
        if month == tenure_months:
            principal_component += balance  # absorb rounding so it closes at 0
            balance = 0.0
        schedule.append({
            "month": month,
            "emi": round(emi, 2),
            "principal_component": round(principal_component, 2),
            "interest_component": round(interest_component, 2),
            "closing_balance": round(max(balance, 0.0), 2),
        })
    total_payment = emi * tenure_months
    return json.dumps({
        "principal": principal, "rate": rate, "tenure_months": tenure_months,
        "emi": round(emi, 2), "total_interest": round(total_payment - principal, 2),
        "total_repayment": round(total_payment, 2), "amortisation_schedule": schedule,
    })


# ---------------------------------------------------------------------------
# 18-19. Gold
# ---------------------------------------------------------------------------

def get_gold_rate(purity_karat: Literal[18, 22, 24]) -> str:
    """Return today's gold rate per gram for a given purity."""
    rate = GOLD_RATES_PER_GRAM.get(purity_karat)
    if rate is None:
        return _error("VALIDATION_ERROR", f"Unsupported purity {purity_karat}")
    return json.dumps({
        "purity_karat": purity_karat, "gold_rate_per_gram": rate, "currency": "INR",
        "date": _today_str(), "source": "MCX",
        "valid_till": (_now() + timedelta(hours=8)).isoformat(),
    })


def calculate_gold_loan_ltv(gold_grams: float, purity_karat: Literal[18, 22, 24], gold_rate_per_gram: float) -> str:
    """Value gold collateral (grams x rate) and apply the bank's current LTV cap
    to compute the maximum eligible loan amount. The LTV cap is applied
    internally by the backend."""
    gross_value = gold_grams * gold_rate_per_gram
    eligible = gross_value * GOLD_LTV_CAP / 100
    return json.dumps({
        "gold_grams": gold_grams, "purity_karat": purity_karat,
        "gold_rate_per_gram": gold_rate_per_gram, "gross_value": round(gross_value, 2),
        "ltv_cap_percent": GOLD_LTV_CAP, "max_eligible_loan_amount": round(eligible, 2),
    })


# ---------------------------------------------------------------------------
# 20-23. Cards
# ---------------------------------------------------------------------------

def get_card_details(card_ids: list[str]) -> str:
    """Return card metadata, status, credit/available limits, and per-channel
    usage controls. Use to diagnose card declines."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    cards = cust.get("cards", {})
    out, errors = [], []
    for cid in card_ids:
        card = cards.get(cid)
        if not card or cid not in ctx["linked_cards"]:
            errors.append({"card_id": cid, "error": "Card not linked to logged-in customer"})
            continue
        out.append(card)
    return json.dumps({"cards": out, "errors": errors})


def toggle_card_freeze(card_id: str, state: Literal["freeze", "unfreeze"]) -> str:
    """Temporarily freeze or unfreeze a card. Reversible. Cannot unfreeze a card
    whose status is blocked or expired."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    card = cust.get("cards", {}).get(card_id)
    if not card or card_id not in ctx["linked_cards"]:
        return _error("VALIDATION_ERROR", f"Card {card_id} not linked to logged-in customer")
    status = card.get("status")
    if status in ("blocked", "expired"):
        return _error("INVALID_STATE", f"Cannot {state} a card whose status is {status}")
    card["status"] = "frozen" if state == "freeze" else "active"
    _persist_db()
    return json.dumps({
        "status": "UPDATED", "card_id": card_id, "action": state,
        "new_card_status": card["status"], "updated_at": _now().isoformat(),
    })


def block_card(card_id: str, reason: Literal["lost", "stolen", "fraud", "damaged"]) -> str:
    """Permanently and irreversibly block a card. A blocked card cannot be
    unblocked; request a replacement via raise_request."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    card = cust.get("cards", {}).get(card_id)
    if not card or card_id not in ctx["linked_cards"]:
        return _error("VALIDATION_ERROR", f"Card {card_id} not linked to logged-in customer")
    if card.get("status") == "blocked":
        return _error("ALREADY_BLOCKED", f"Card {card_id} is already blocked")
    card["status"] = "blocked"
    card["block_reason"] = reason
    _persist_db()
    return json.dumps({
        "status": "BLOCKED", "card_id": card_id, "reason": reason, "reversible": False,
        "note": "Card permanently blocked. Raise a service request for a replacement.",
        "blocked_at": _now().isoformat(),
    })


def set_card_controls(
    card_id: str,
    atm: Optional[dict[str, Any]] = None,
    online: Optional[dict[str, Any]] = None,
    pos: Optional[dict[str, Any]] = None,
) -> str:
    """Update per-channel card controls. Each channel (atm, online, pos) has
    independent domestic and international enable/limit settings. Only provided
    fields change. A daily_limit supplied with enabled: false is ignored."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    card = cust.get("cards", {}).get(card_id)
    if not card or card_id not in ctx["linked_cards"]:
        return _error("VALIDATION_ERROR", f"Card {card_id} not linked to logged-in customer")
    controls = card.setdefault("controls", {})
    for channel, payload in (("atm", atm), ("online", online), ("pos", pos)):
        if payload is None:
            continue
        chan_ctrl = controls.setdefault(channel, {})
        for region in ("domestic", "international"):
            region_payload = payload.get(region)
            if region_payload is None:
                continue
            enabled = bool(region_payload.get("enabled"))
            new_region = {"enabled": enabled}
            # daily_limit stored only when enabled is true.
            if enabled and "daily_limit" in region_payload:
                new_region["daily_limit"] = region_payload["daily_limit"]
            chan_ctrl[region] = new_region
    _persist_db()
    return json.dumps({
        "status": "UPDATED", "card_id": card_id, "controls": controls,
        "updated_at": _now().isoformat(),
    })


# ---------------------------------------------------------------------------
# 24-25. Mandates
# ---------------------------------------------------------------------------

def show_mandates(
    account_id: Optional[str] = None,
    status: Literal["active", "paused", "cancelled", "all"] = "all",
) -> str:
    """List standing mandates (e-NACH, UPI AutoPay, standing instructions) on the
    customer's profile."""
    cust, err = _active_customer()
    if err:
        return err
    mandates = list(cust.get("mandates", []))
    if account_id:
        mandates = [m for m in mandates if m.get("account_id") == account_id]
    if status != "all":
        mandates = [m for m in mandates if m.get("status") == status]
    return json.dumps({"total": len(mandates), "mandates": mandates})


def cancel_mandate(mandate_id: str) -> str:
    """Permanently cancel a mandate, stopping all future debits under it.
    Already-processed debits are unaffected."""
    cust, err = _active_customer()
    if err:
        return err
    for mandate in cust.get("mandates", []):
        if mandate.get("mandate_id") == mandate_id:
            if mandate.get("status") == "cancelled":
                return _error("ALREADY_CANCELLED", f"Mandate {mandate_id} is already cancelled")
            mandate["status"] = "cancelled"
            mandate["cancelled_on"] = _today_str()
            _persist_db()
            return json.dumps({
                "status": "CANCELLED", "mandate_id": mandate_id, "payee": mandate.get("payee"),
                "future_debits_stopped": True, "cancelled_at": _now().isoformat(),
            })
    return _error("VALIDATION_ERROR", f"Mandate {mandate_id} not found for logged-in customer")


# ---------------------------------------------------------------------------
# 26-27. Cheques
# ---------------------------------------------------------------------------

def stop_cheque_payment(
    account_id: str,
    cheque_numbers: list[str],
    reason: Optional[Literal["lost", "dispute", "other"]] = None,
) -> str:
    """Place a stop-payment instruction on one or more cheques issued from an
    account. Only affects cheques that have not already cleared."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    acc = cust.get("accounts", {}).get(account_id)
    if not acc or account_id not in ctx["linked_accounts"]:
        return _error("VALIDATION_ERROR", f"Account {account_id} not linked to logged-in customer")
    cheques = {c.get("cheque_number"): c for c in acc.get("cheques", [])}
    results = []
    for num in cheque_numbers:
        cheque = cheques.get(num)
        if cheque is None:
            # Unknown cheque number → still record a stop instruction optimistically.
            results.append({"cheque_number": num, "outcome": "STOP_PLACED", "note": "no record found; stop registered"})
            acc.setdefault("cheques", []).append({"cheque_number": num, "status": "stopped"})
        elif cheque.get("status") == "cleared":
            results.append({"cheque_number": num, "outcome": "CANNOT_STOP", "note": "cheque already cleared"})
        else:
            cheque["status"] = "stopped"
            results.append({"cheque_number": num, "outcome": "STOP_PLACED"})
    _persist_db()
    return json.dumps({
        "account_id": account_id, "reason": reason or "other",
        "reference": _ref("STP", 8), "results": results,
    })


def request_cheque_book(
    account_id: str,
    leaves: int = 25,
    delivery_address: Optional[str] = None,
) -> str:
    """Order a new cheque book for a savings or current account."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    if account_id not in ctx["linked_accounts"]:
        return _error("VALIDATION_ERROR", f"Account {account_id} not linked to logged-in customer")
    return json.dumps({
        "status": "REQUESTED", "request_id": _ref("CHQ", 8), "account_id": account_id,
        "leaves": leaves, "delivery_address": delivery_address or "registered address",
        "expected_delivery": (_now() + timedelta(days=7)).strftime("%Y-%m-%d"),
        "requested_at": _now().isoformat(),
    })


# ---------------------------------------------------------------------------
# 28. Duplicate statement
# ---------------------------------------------------------------------------

def request_duplicate_statement(
    account_id: str,
    from_date: str,
    to_date: str,
    delivery_mode: Literal["email", "download"] = "email",
) -> str:
    """Order an official duplicate account statement for a date range, delivered
    by email or made available for download."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    if account_id not in ctx["linked_accounts"]:
        return _error("VALIDATION_ERROR", f"Account {account_id} not linked to logged-in customer")
    ref = _ref("STMT", 8)
    resp: dict[str, Any] = {
        "status": "REQUESTED", "request_id": ref, "account_id": account_id,
        "from_date": from_date, "to_date": to_date, "delivery_mode": delivery_mode,
        "requested_at": _now().isoformat(),
    }
    if delivery_mode == "email":
        resp["delivered_to"] = cust.get("profile", {}).get("email", "registered email")
    else:
        resp["download_url"] = f"https://bank.example.com/statements/{ref}.pdf"
        resp["expiry"] = (_now() + timedelta(days=7)).isoformat()
    return json.dumps(resp)


# ---------------------------------------------------------------------------
# 29. Address update
# ---------------------------------------------------------------------------

def update_address(
    address_type: Literal["communication", "permanent"],
    line1: str,
    city: str,
    state: str,
    pincode: str,
    line2: Optional[str] = None,
) -> str:
    """Log a request to update the communication or permanent address. The change
    is effective only after bank verification — never state it is already
    updated."""
    cust, err = _active_customer()
    if err:
        return err
    requested = {"line1": line1, "line2": line2, "city": city, "state": state, "pincode": pincode}
    cust.setdefault("pending_address_updates", []).append({
        "address_type": address_type, "address": requested,
        "requested_at": _now().isoformat(), "status": "PENDING_VERIFICATION",
    })
    _persist_db()
    return json.dumps({
        "status": "REQUEST_LOGGED", "request_id": _ref("ADR", 8),
        "address_type": address_type, "requested_address": requested,
        "verification_status": "PENDING_VERIFICATION",
        "note": "Request logged; change is effective only after bank verification.",
        "logged_at": _now().isoformat(),
    })


# ---------------------------------------------------------------------------
# 30-31. Service requests
# ---------------------------------------------------------------------------

def raise_request(
    description: str,
    category: Literal["failed_transaction", "unauthorized_debit", "service_quality",
                      "app_issue", "charges_dispute", "general_query", "other"] = "general_query",
    related_transaction_id: Optional[str] = None,
    account_id: Optional[str] = None,
) -> str:
    """Register a complaint or general service request — the single escalation
    channel for disputes, service issues, and any action no other tool can
    perform.

    Agent-only: infer `category` and `related_transaction_id` from the
    conversation. Do not ask the customer for backend field names or category
    codes (e.g. unauthorized_debit); confirm the facts in plain language first.
    """
    cust, err = _active_customer()
    if err:
        return err
    if category in ("failed_transaction", "unauthorized_debit") and not related_transaction_id:
        return _error("VALIDATION_ERROR",
                      f"related_transaction_id is required for category {category}")
    urgent = category == "unauthorized_debit"
    request_id = _ref("SR-", 8)
    eta_days = 1 if urgent else 5
    record = {
        "request_id": request_id, "category": category, "description": description,
        "related_transaction_id": related_transaction_id, "account_id": account_id,
        "status": "OPEN", "priority": "URGENT" if urgent else "NORMAL",
        "created_at": _now().isoformat(),
        "eta": (_now() + timedelta(days=eta_days)).strftime("%Y-%m-%d"),
    }
    cust.setdefault("requests", {})[request_id] = record
    _persist_db()
    return json.dumps({
        "status": "RAISED", "request_id": request_id, "category": category,
        "priority": record["priority"], "eta": record["eta"],
        "note": "Request routed for review; no specific outcome is promised.",
        "created_at": record["created_at"],
    })


def get_request_status(request_id: str) -> str:
    """Return the current status of a previously raised service request."""
    cust, err = _active_customer()
    if err:
        return err
    record = cust.get("requests", {}).get(request_id)
    if not record:
        return _error("VALIDATION_ERROR", f"Service request {request_id} not found")
    return json.dumps({
        "request_id": request_id, "category": record.get("category"),
        "status": record.get("status", "OPEN"), "priority": record.get("priority"),
        "description": record.get("description"), "created_at": record.get("created_at"),
        "eta": record.get("eta"),
    })


# ---------------------------------------------------------------------------
# 32. Insurance
# ---------------------------------------------------------------------------

def get_insurance_details(policy_ids: list[str]) -> str:
    """Return policy details for the customer's bancassurance policies. No
    premium-payment or claim tools exist — route those via raise_request."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    policies = cust.get("insurance", {})
    out, errors = [], []
    for pid in policy_ids:
        pol = policies.get(pid)
        if not pol or pid not in ctx["linked_products"]:
            errors.append({"policy_id": pid, "error": "Policy not linked to logged-in customer"})
            continue
        out.append(pol)
    return json.dumps({"policies": out, "errors": errors,
                       "note": "Premium payment and claims are handled via raise_request."})


# ---------------------------------------------------------------------------
# 33. Products & offers
# ---------------------------------------------------------------------------

def get_products_and_offers(
    type: Literal["deposit", "loan", "card", "insurance", "offer", "all"] = "all",
) -> str:
    """Return a combined list of the bank's products and the customer's
    personalised offers, optionally filtered by type. Offers are indicative —
    never state one is guaranteed. Never quote rates from here; use
    get_deposit_loan_rates."""
    cust, _ = _active_customer()
    products = PRODUCT_CATALOG
    offers = cust.get("offers", []) if cust else []
    if type == "offer":
        products = []
    elif type != "all":
        products = [p for p in products if p["type"] == type]
        offers = [o for o in offers if o.get("type") == type]
    return json.dumps({
        "filter": type,
        "products": products,
        "offers": offers,
        "disclaimer": "Offers are indicative and subject to eligibility verification. "
                      "Rates must be confirmed via get_deposit_loan_rates.",
    })
