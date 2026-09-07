"""
Talk-to-Data Production Orchestrator Agent for NeoStats Credit Risk Intelligence Platform.

Coordinates the end-to-end conversational analytics pipeline:
1. Natural language question ingestion
2. Debounce event suppression (400ms) & thread execution lock
3. Thread-safe in-memory LRU cache retrieval (last 50 queries)
4. Dual Token-Bucket rate limit acquisition (RPM & TPM)
5. Multi-tier LLM cascade SQL routing (Groq primary/secondary -> Ollama -> Deterministic)
6. AST-based SQL security validation (sqlglot read-only SELECT guardrail)
7. Strictly read-only execution in DuckDB (data/credit_risk.duckdb)
8. Executive CRO business summary generation
9. Cache population and structured result delivery
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from src.talk_to_data.debounce import (
    DebounceSuppressionError,
    Debouncer,
    QueryLRUCache,
)
from src.talk_to_data.prompts import BENCHMARK_QUERIES
from src.talk_to_data.rate_limiter import (
    RateLimitExceededError,
    TokenBucketRateLimiter,
)
from src.talk_to_data.router import ModelRouter
from src.talk_to_data.validator import (
    SecurityValidationError,
    check_query_scope,
    validate_sql,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Structured response container returned by TalkToDataAgent."""

    question: str
    sql: str
    data: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    summary: str
    sql_tier: str
    summary_tier: str
    cache_hit: bool
    execution_time_ms: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert returned tabular data to a pandas DataFrame."""
        if not self.data:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(self.data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize response object to dictionary."""
        return asdict(self)

    def format_markdown(self) -> str:
        """Format the complete output as a comprehensive Markdown report."""
        if not self.data:
            table_md = "_No records returned._"
        else:
            headers = self.columns
            lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in self.data[:15]:
                lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
            table_md = "\n".join(lines)

        cache_indicator = "[CACHE HIT]" if self.cache_hit else "[FRESH EXECUTION]"

        return (
            f"## Talk-to-Data Risk Intelligence Report {cache_indicator}\n\n"
            f"**User Question:** {self.question}\n\n"
            f"**Execution Tier:** `{self.sql_tier}` | **Summary Tier:** `{self.summary_tier}`\n"
            f"**Latency:** {self.execution_time_ms:.1f}ms | **Rows:** {self.row_count}\n\n"
            f"### Executed DuckDB SQL\n```sql\n{self.sql}\n```\n\n"
            f"### Data Results Preview\n{table_md}\n\n"
            f"{self.summary}\n"
        )


class TalkToDataAgent:
    """
    Production conversational data engine for credit risk portfolio queries.
    Enforces security, rate limiting, debouncing, multi-tier fallback, and read-only execution.
    """

    DEFAULT_DB_PATH = "data/credit_risk.duckdb"
    DEFAULT_CSV_PATH = "data/sample_application.csv"

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        csv_path: str = DEFAULT_CSV_PATH,
        max_rpm: int = 30,
        max_tpm: int = 60_000,
        debounce_ms: float = 400.0,
        cache_capacity: int = 50,
        groq_api_key: Optional[str] = None,
        auto_init_db: bool = True,
    ) -> None:
        """
        Initialize the TalkToDataAgent and underlying engines.

        Args:
            db_path: Path to DuckDB database file.
            csv_path: Path to source CSV for automatic database bootstrapping.
            max_rpm: Requests-per-minute rate limit.
            max_tpm: Tokens-per-minute rate limit.
            debounce_ms: Debounce window in milliseconds.
            cache_capacity: Maximum items stored in LRU cache.
            groq_api_key: Optional override for Groq API key.
            auto_init_db: If True, ensures database and table exist before executing.
        """
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path)

        # Core subsystems
        self.rate_limiter = TokenBucketRateLimiter(max_rpm=max_rpm, max_tpm=max_tpm)
        self.debouncer = Debouncer(suppression_ms=debounce_ms)
        self.cache = QueryLRUCache(capacity=cache_capacity)
        self.router = ModelRouter(groq_api_key=groq_api_key, auto_verify=False)

        if auto_init_db:
            self._ensure_database_initialized()

    def _ensure_database_initialized(self) -> None:
        """Bootstrap the DuckDB database and loan_applications table if missing."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect with write access temporarily only for bootstrapping schema
        conn = duckdb.connect(str(self.db_path), read_only=False)
        try:
            # Check if loan_applications table exists
            tables = conn.execute("SHOW TABLES;").fetchall()
            table_names = [t[0].lower() for t in tables]

            if "loan_applications" not in table_names:
                if self.csv_path.exists():
                    logger.info("Initializing 'loan_applications' table in %s from %s", self.db_path, self.csv_path)
                    conn.execute(
                        f"CREATE TABLE loan_applications AS SELECT * FROM read_csv_auto('{self.csv_path}')"
                    )
                else:
                    logger.warning("Source CSV %s not found. Table initialization skipped.", self.csv_path)
        finally:
            conn.close()

    def _execute_read_only_sql(self, sql_query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Execute SQL query in DuckDB strictly in read-only mode.

        Args:
            sql_query: Sanitized SQL string.

        Returns:
            Tuple of (list of row dictionaries, list of column names).
        """
        # Open in strictly read-only mode
        conn = duckdb.connect(str(self.db_path), read_only=True)
        try:
            cursor = conn.execute(sql_query)
            description = cursor.description or []
            columns = [col[0] for col in description]
            rows = cursor.fetchall()
            dict_rows = [dict(zip(columns, row)) for row in rows]
            return dict_rows, columns
        finally:
            conn.close()

    def ask(
        self,
        question: str,
        session_id: str = "default",
        bypass_cache: bool = False,
    ) -> AgentResponse:
        """
        End-to-end query workflow from natural language to validated SQL and executive briefing.

        Args:
            question: Natural language question on credit risk data.
            session_id: Session/user identifier for debouncing.
            bypass_cache: Force fresh model generation and query execution.

        Returns:
            AgentResponse containing SQL, executed dataset, executive summary, and latency.
        """
        start_time = time.perf_counter()
        clean_q = question.strip()

        # -------------------------------------------------------------
        # 0. Domain Scope and Adversarial Guardrail Validation
        # -------------------------------------------------------------
        is_in_scope, refusal_reason = check_query_scope(clean_q)
        if not is_in_scope:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            scope_message = (
                "### Domain Scope Notice\n\n"
                f"**Refusal Reason:** {refusal_reason}\n\n"
                "The Talk-to-Data conversational assistant is strictly restricted to credit risk analytics, loan portfolio inquiries, and underwriting intelligence.\n\n"
                "**Permitted In-Scope Topics:**\n"
                "- Portfolio default rates and risk distributions (`TARGET = 0` vs `1`)\n"
                "- Applicant demographics: age, education, occupation, income type\n"
                "- Financial debt ratios: credit amount (`AMT_CREDIT`), income (`AMT_INCOME_TOTAL`), annuity (`AMT_ANNUITY`)\n"
                "- External bureau ratings (`EXT_SOURCE_1/2/3`) and credit inquiries\n"
                "- Delinquencies and social circle defaults (`DEF_30`, `DEF_60`)\n\n"
                "Please rephrase your question to focus on credit risk or loan portfolio analytics."
            )
            return AgentResponse(
                question=clean_q,
                sql="",
                data=[],
                columns=[],
                row_count=0,
                summary=scope_message,
                sql_tier="Domain Guardrail (Blocked)",
                summary_tier="Domain Guardrail (Refusal)",
                cache_hit=False,
                execution_time_ms=elapsed_ms,
                metadata={"out_of_scope": True, "refusal_reason": refusal_reason},
            )

        # -------------------------------------------------------------
        # 1. Debounce Check (400ms suppression & execution guard)
        # -------------------------------------------------------------
        try:
            self.debouncer.acquire(key=session_id, raise_on_suppress=True)
        except DebounceSuppressionError as exc:
            # Check if we can serve previous result from cache rather than hard failing
            cached = self.cache.get(clean_q)
            if cached:
                logger.info("Debounce active; serving cached response for: %s", clean_q)
                data_rows, cols = self._execute_read_only_sql(cached["sql"])
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return AgentResponse(
                    question=clean_q,
                    sql=cached["sql"],
                    data=data_rows,
                    columns=cols,
                    row_count=len(data_rows),
                    summary=cached["summary"],
                    sql_tier=cached.get("metadata", {}).get("sql_tier", "Cache"),
                    summary_tier=cached.get("metadata", {}).get("summary_tier", "Cache"),
                    cache_hit=True,
                    execution_time_ms=elapsed_ms,
                    metadata={"debounced": True},
                )
            raise exc

        # -------------------------------------------------------------
        # 2. In-Memory LRU Cache Check
        # -------------------------------------------------------------
        if not bypass_cache:
            cached_entry = self.cache.get(clean_q)
            if cached_entry:
                data_rows, cols = self._execute_read_only_sql(cached_entry["sql"])
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return AgentResponse(
                    question=clean_q,
                    sql=cached_entry["sql"],
                    data=data_rows,
                    columns=cols,
                    row_count=len(data_rows),
                    summary=cached_entry["summary"],
                    sql_tier=cached_entry.get("metadata", {}).get("sql_tier", "Cache"),
                    summary_tier=cached_entry.get("metadata", {}).get("summary_tier", "Cache"),
                    cache_hit=True,
                    execution_time_ms=elapsed_ms,
                )

        # -------------------------------------------------------------
        # 3. Dual Token-Bucket Rate Limiter Acquisition
        # -------------------------------------------------------------
        # Estimated token allocation: ~600 tokens per full Q&A turn
        self.rate_limiter.acquire(estimated_tokens=600, wait=True, timeout=12.0)

        # -------------------------------------------------------------
        # 4. Multi-Tier LLM Cascade SQL Generation
        # -------------------------------------------------------------
        raw_sql, sql_tier = self.router.generate_sql(clean_q)

        # Check for model refusal tokens
        if raw_sql.strip().upper().startswith("REFUSAL:") or "OUT_OF_SCOPE" in raw_sql.upper():
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            refusal_body = raw_sql.replace("REFUSAL:", "").replace("OUT_OF_SCOPE:", "").strip()
            return AgentResponse(
                question=clean_q,
                sql="",
                data=[],
                columns=[],
                row_count=0,
                summary=f"### Domain Scope Refusal\n\n{refusal_body}\n\nPlease submit an inquiry concerning credit risk, loan defaults, or borrower financials.",
                sql_tier=sql_tier,
                summary_tier="LLM Domain Guardrail",
                cache_hit=False,
                execution_time_ms=elapsed_ms,
                metadata={"out_of_scope": True, "refusal_reason": refusal_body},
            )

        # -------------------------------------------------------------
        # 5. AST SQL Security Guardrail Validation (sqlglot)
        # -------------------------------------------------------------
        try:
            sanitized_sql = validate_sql(raw_sql, dialect="duckdb")
        except SecurityValidationError as sec_exc:
            logger.error("Generated SQL failed security validation: %s. Falling back to benchmark.", sec_exc)
            # Safe failover to deterministic benchmark query
            fallback_match = self.router.match_deterministic_benchmark(clean_q)
            if fallback_match:
                sanitized_sql = validate_sql(fallback_match["sql"], dialect="duckdb")
                sql_tier = f"Tier 4: Deterministic Guardrail Recovery (Benchmark #{fallback_match['id']})"
            else:
                raise sec_exc

        # -------------------------------------------------------------
        # 6. Read-Only DuckDB Execution
        # -------------------------------------------------------------
        data_rows, columns = self._execute_read_only_sql(sanitized_sql)

        # Create quick markdown text preview for summary generation
        if data_rows:
            preview_df = pd.DataFrame(data_rows).head(10)
            table_preview = preview_df.to_string(index=False)
        else:
            table_preview = "No rows returned."

        # -------------------------------------------------------------
        # 7. Executive CRO Business Summary Generation
        # -------------------------------------------------------------
        summary_text, summary_tier = self.router.generate_summary(
            clean_q, sanitized_sql, table_preview, data_rows=data_rows
        )

        # -------------------------------------------------------------
        # 8. Update In-Memory LRU Cache
        # -------------------------------------------------------------
        meta = {
            "sql_tier": sql_tier,
            "summary_tier": summary_tier,
            "row_count": len(data_rows),
        }
        self.cache.put(clean_q, sanitized_sql, summary_text, metadata=meta)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return AgentResponse(
            question=clean_q,
            sql=sanitized_sql,
            data=data_rows,
            columns=columns,
            row_count=len(data_rows),
            summary=summary_text,
            sql_tier=sql_tier,
            summary_tier=summary_tier,
            cache_hit=False,
            execution_time_ms=elapsed_ms,
            metadata=meta,
        )
