"""
NeoStats Talk-to-Data Conversational Analytics Package.

Exposes production components for:
- Token-Bucket Rate Limiting (rate_limiter)
- Debounce & LRU Caching (debounce)
- AST-based SQL Security Guardrails (validator)
- DuckDB Schema & Few-shot Benchmark Prompts (prompts)
- Multi-tier LLM Cascaded Routing (router)
- End-to-End Orchestrator Agent (agent)
"""

from src.talk_to_data.agent import AgentResponse, TalkToDataAgent
from src.talk_to_data.debounce import (
    DebounceSuppressionError,
    Debouncer,
    QueryLRUCache,
)
from src.talk_to_data.prompts import (
    BENCHMARK_QUERIES,
    DUCKDB_SCHEMA_DEFINITION,
    SYSTEM_PROMPT_EXECUTIVE_SUMMARY,
    SYSTEM_PROMPT_SQL,
)
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

__all__ = [
    "TalkToDataAgent",
    "AgentResponse",
    "TokenBucketRateLimiter",
    "RateLimitExceededError",
    "execute_with_backoff",
    "Debouncer",
    "DebounceSuppressionError",
    "QueryLRUCache",
    "validate_sql",
    "extract_sql_from_markdown",
    "SecurityValidationError",
    "ModelRouter",
    "BENCHMARK_QUERIES",
    "DUCKDB_SCHEMA_DEFINITION",
    "SYSTEM_PROMPT_SQL",
    "SYSTEM_PROMPT_EXECUTIVE_SUMMARY",
]
