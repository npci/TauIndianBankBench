"""The knowledge-base tool must expose `query` and search the markdown corpus."""

import inspect
import json

from tau2.domains.indian_banking.environment import get_environment
from tau2.domains.indian_banking.service import search_knowledge_base

RESULT_KEYS = {
    "text",
    "source_url",
    "source_date",
    "bank_id",
    "category",
    "label",
    "relevance",
    "score",
    "similarity",
}


def test_query_is_the_first_parameter():
    parameters = inspect.signature(search_knowledge_base).parameters
    assert list(parameters) == ["query", "bank_id", "category", "top_k"]
    assert parameters["query"].default is inspect.Parameter.empty
    for optional in ("bank_id", "category", "top_k"):
        assert parameters[optional].default is None


def test_tool_schema_exposes_query():
    env = get_environment()
    schema = env.tools.get_tools()["search_knowledge_base"].openai_schema
    properties = schema["function"]["parameters"].get("properties") or {}
    assert "query" in properties


def test_search_returns_scored_results():
    data = json.loads(search_knowledge_base("gold loan LTV maximum"))
    assert data["query"] == "gold loan LTV maximum"
    assert data["total_results"] >= 1
    for result in data["results"]:
        assert RESULT_KEYS <= set(result)
        assert 0.0 <= result["score"] <= 1.0
        assert result["similarity"] == result["score"]
        assert result["text"]


def test_top_k_limits_results():
    data = json.loads(search_knowledge_base("UPI transaction limit", top_k=2))
    assert len(data["results"]) <= 2
    assert data["total_results"] == len(data["results"])


def test_category_narrows_the_corpus():
    data = json.loads(search_knowledge_base("interest", category="Accounts"))
    assert data["results"]
    assert all("accounts" in r["category"].lower() for r in data["results"])


def test_search_is_deterministic():
    first = search_knowledge_base("premature withdrawal penalty")
    second = search_knowledge_base("premature withdrawal penalty")
    assert first == second


def test_unmatched_query_still_returns_something():
    data = json.loads(search_knowledge_base("zzzz nothing matches here"))
    assert data["total_results"] >= 1
