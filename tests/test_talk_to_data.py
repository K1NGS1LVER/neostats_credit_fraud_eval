"""
Comprehensive test suite for the Talk-to-Data engine.
Tests rate limiting, debounce suppression, LRU cache, AST SQL security,
model router cascades, and end-to-end agent orchestration.
"""

import time
import pytest
import duckdb

from src.talk_to_data.agent import TalkToDataAgent, AgentResponse
from src.talk_to_data.debounce import (
    DebounceSuppressionError,
    Debouncer,
    QueryLRUCache,
)
from src.talk_to_data.prompts import BENCHMARK_QUERIES
from src.talk_to_data.rate_limiter import (
    RateLimitExceededError,
    TokenBucketRateLimiter,
    execute_with_backoff,
)
from src.talk_to_data.router import ModelRouter
from src.talk_to_data.validator import (
    SecurityValidationError,
    extract_sql_from_markdown,
    validate_sql,
)


# =====================================================================
# 1. Rate Limiter Tests
# =====================================================================

def test_rate_limiter_dual_bucket():
    limiter = TokenBucketRateLimiter(max_rpm=60, max_tpm=6000)
    assert limiter.acquire(estimated_tokens=100, wait=False) is True
    headroom = limiter.get_headroom()
    assert headroom["rpm_available"] < 60
    assert headroom["tpm_available"] < 6000
    assert headroom["has_capacity"] is True


def test_rate_limiter_exhaustion():
    limiter = TokenBucketRateLimiter(max_rpm=1, max_tpm=100)
    assert limiter.acquire(estimated_tokens=50, wait=False) is True
    with pytest.raises(RateLimitExceededError):
        limiter.acquire(estimated_tokens=50, wait=False)


def test_rate_limiter_backoff_retry_with_jitter():
    attempts = 0

    def mock_flaky_api():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            err = Exception("Rate limit exceeded: please try again in 0.05s")
            err.status_code = 429
            raise err
        return "SUCCESS_DATA"

    result = execute_with_backoff(mock_flaky_api, max_retries=3, initial_delay=0.01)
    assert result == "SUCCESS_DATA"
    assert attempts == 3


# =====================================================================
# 2. Debounce and LRU Cache Tests
# =====================================================================

def test_debouncer_event_suppression():
    debouncer = Debouncer(suppression_ms=400.0)
    assert debouncer.acquire("session1") is True

    # Immediate second click must be suppressed
    with pytest.raises(DebounceSuppressionError):
        debouncer.acquire("session1", raise_on_suppress=True)

    assert debouncer.acquire("session1", raise_on_suppress=False) is False

    # Wait for cooldown to expire
    time.sleep(0.42)
    assert debouncer.acquire("session1") is True


def test_debouncer_execution_guard_context_manager():
    debouncer = Debouncer(suppression_ms=100.0)
    with debouncer.execution_guard("keyA"):
        pass

    # Rapid call within 100ms
    with pytest.raises(DebounceSuppressionError):
        with debouncer.execution_guard("keyA"):
            pass


def test_lru_cache_50_capacity():
    cache = QueryLRUCache(capacity=50)
    for i in range(60):
        cache.put(f"query_{i}", f"SELECT {i}", f"Summary {i}")

    assert cache.size() == 50
    # First 10 items (0..9) should be evicted
    assert cache.get("query_0") is None
    assert cache.get("query_9") is None
    assert cache.get("query_59") is not None


def test_lru_cache_mru_promotion():
    cache = QueryLRUCache(capacity=3)
    cache.put("q1", "SQL 1", "Sum 1")
    cache.put("q2", "SQL 2", "Sum 2")
    cache.put("q3", "SQL 3", "Sum 3")

    # Access q1 to promote it
    item = cache.get("q1")
    assert item["sql"] == "SQL 1"

    # Insert q4 -> should evict q2 (oldest non-promoted)
    cache.put("q4", "SQL 4", "Sum 4")
    assert cache.get("q2") is None
    assert cache.get("q1") is not None
    assert cache.get("q3") is not None
    assert cache.get("q4") is not None


# =====================================================================
# 3. SQL Validator Guardrail Tests
# =====================================================================

def test_validator_single_select_enforcement():
    valid = "SELECT NAME_EDUCATION_TYPE, COUNT(*) FROM loan_applications GROUP BY NAME_EDUCATION_TYPE"
    sanitized = validate_sql(valid)
    assert sanitized.upper().startswith("SELECT")


def test_validator_markdown_extraction():
    md = "```sql\nSELECT * FROM loan_applications LIMIT 5;\n```"
    extracted = extract_sql_from_markdown(md)
    assert "SELECT * FROM loan_applications LIMIT 5" in extracted
    sanitized = validate_sql(md)
    assert "loan_applications" in sanitized


def test_validator_blocks_chaining():
    chained = "SELECT * FROM loan_applications; SELECT * FROM loan_applications;"
    with pytest.raises(SecurityValidationError, match="Multiple SQL statements detected|Query chaining"):
        validate_sql(chained)


@pytest.mark.parametrize(
    "destructive_sql",
    [
        "DROP TABLE loan_applications;",
        "DELETE FROM loan_applications WHERE TARGET = 1;",
        "UPDATE loan_applications SET TARGET = 0;",
        "INSERT INTO loan_applications VALUES (1);",
        "ALTER TABLE loan_applications DROP COLUMN TARGET;",
        "CREATE TABLE hack AS SELECT * FROM loan_applications;",
        "TRUNCATE TABLE loan_applications;",
    ],
)
def test_validator_blocks_ddl_dml(destructive_sql):
    with pytest.raises(SecurityValidationError):
        validate_sql(destructive_sql)


def test_validator_blocks_non_select_root():
    with pytest.raises(SecurityValidationError, match="Root expression must be a SELECT"):
        validate_sql("VACUUM FULL;")


# =====================================================================
# 4. Router and Deterministic Cascade Tests
# =====================================================================

def test_deterministic_matcher_all_5_benchmarks():
    router = ModelRouter(groq_api_key="", auto_verify=False, ollama_endpoint="http://invalid:9999")

    benchmark_inputs = [
        "Average default rate by education level",
        "Top 5 occupations by average credit amount",
        "Default rate comparison between male and female applicants across income brackets",
        "Percentage of high-risk applicants with external scores below 0.3",
        "Average credit-to-income ratio for approved vs defaulted applicants",
    ]

    for idx, prompt_text in enumerate(benchmark_inputs, 1):
        matched = router.match_deterministic_benchmark(prompt_text)
        assert matched is not None, f"Failed to match benchmark #{idx}: {prompt_text}"
        assert matched["id"] == idx

        sql, tier = router.generate_sql(prompt_text)
        assert "Tier 4" in tier
        assert f"Benchmark #{idx}" in tier
        # Ensure it passes security validation
        validate_sql(sql)


def test_router_deterministic_summary_generation():
    router = ModelRouter(groq_api_key="", auto_verify=False, ollama_endpoint="http://invalid:9999")
    for b in BENCHMARK_QUERIES:
        summary, tier = router.generate_summary(
            b["question"], b["sql"], "Preview Table", data_rows=[{"education_level": "Higher", "default_rate_pct": 5.2}]
        )
        assert "Tier 4" in tier
        assert "Key Metric Finding" in summary
        assert "Credit Risk Implication" in summary
        assert "Recommended Strategic Action" in summary


# =====================================================================
# 5. Agent End-to-End Orchestrator Tests
# =====================================================================

def test_agent_orchestrator_offline_mode():
    agent = TalkToDataAgent(
        groq_api_key="",  # force offline to test Tier 4 deterministic fallback
        debounce_ms=50.0,
    )

    # Ask Benchmark 1
    resp1 = agent.ask("Average default rate by education level")
    assert isinstance(resp1, AgentResponse)
    assert resp1.row_count > 0
    assert resp1.cache_hit is False
    assert "Tier 4" in resp1.sql_tier
    assert len(resp1.data) == 5

    # Check cache hit on repeat
    time.sleep(0.06)
    resp2 = agent.ask("Average default rate by education level")
    assert resp2.cache_hit is True
    assert resp2.row_count == resp1.row_count

    # Verify DataFrame export
    df = resp1.to_dataframe()
    assert len(df) == 5
    assert "education_level" in df.columns or "NAME_EDUCATION_TYPE" in df.columns or "default_rate_pct" in df.columns

    # Verify Markdown formatting
    md_report = resp1.format_markdown()
    assert "Talk-to-Data Risk Intelligence Report" in md_report
    assert "Executed DuckDB SQL" in md_report
