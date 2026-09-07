"""
AST-based SQL Security Guardrail Module using sqlglot.

Strictly enforces:
1. Only 1 SQL statement (prohibits query chaining via semicolons).
2. Root expression is exp.Select.
3. Explicitly blocks any destructive DDL/DML: Drop, Delete, Update, Insert, Alter, Create, Truncate.
4. Returns sanitized SQL or raises a descriptive SecurityValidationError.
"""

from __future__ import annotations

import re
from typing import Sequence, Tuple, Type
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, SqlglotError


class SecurityValidationError(ValueError):
    """Raised when an incoming SQL query violates read-only AST security rules."""
    pass


# Explicit destructive DDL/DML AST nodes blocked across all sub-expressions
FORBIDDEN_EXPRESSION_TYPES: Tuple[Type[exp.Expression], ...] = (
    exp.Drop,
    exp.Delete,
    exp.Update,
    exp.Insert,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Command,
    exp.Merge,
    exp.Pragma,
)

# Additional keyword checks to guard against raw unparsed command execution or dialect bypass
BLOCKED_KEYWORDS = (
    r"\bdrop\b",
    r"\bdelete\b",
    r"\bupdate\b",
    r"\binsert\b",
    r"\balter\b",
    r"\bcreate\b",
    r"\btruncate\b",
    r"\bcopy\s+to\b",
    r"\binstall\b",
    r"\bload\b",
    r"\battach\b",
    r"\bdetach\b",
    r"\bpragma\b",
    r"\bshutdown\b",
)


def extract_sql_from_markdown(raw_query: str) -> str:
    """
    Extract clean SQL statement from LLM response text, stripping markdown code blocks.

    Args:
        raw_query: Raw LLM output string, possibly enclosed in markdown fences.

    Returns:
        Cleaned SQL string.
    """
    cleaned = raw_query.strip()

    # Match ```sql ... ``` or ``` ... ```
    match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()

    # Remove any dangling commentary before or after if line begins with SELECT or WITH
    lines = cleaned.splitlines()
    filtered_lines = []
    found_sql_start = False
    for line in lines:
        stripped = line.strip()
        if not found_sql_start:
            if re.match(r"^(SELECT|WITH)\b", stripped, re.IGNORECASE):
                found_sql_start = True
                filtered_lines.append(line)
        else:
            filtered_lines.append(line)

    if found_sql_start:
        cleaned = "\n".join(filtered_lines).strip()

    return cleaned


def validate_sql(raw_sql: str, dialect: str = "duckdb") -> str:
    """
    Validate that an incoming SQL query satisfies all strict security criteria:
    1. Exactly 1 SQL statement.
    2. Root AST expression is an instance of exp.Select.
    3. Contains no destructive DDL/DML expressions (Drop, Delete, Update, Insert, Alter, Create, Truncate).

    Args:
        raw_sql: The SQL string to validate and sanitize.
        dialect: The SQL dialect (default: "duckdb").

    Returns:
        Sanitized, canonicalized SQL string formatted for DuckDB.

    Raises:
        SecurityValidationError: If the query violates any security guardrails.
    """
    if not raw_sql or not raw_sql.strip():
        raise SecurityValidationError("SQL statement is empty.")

    cleaned_sql = extract_sql_from_markdown(raw_sql)

    # 1. Parse into AST expressions
    try:
        parsed_statements: Sequence[exp.Expression | None] = sqlglot.parse(
            cleaned_sql, read=dialect
        )
    except ParseError as exc:
        raise SecurityValidationError(f"SQL parsing syntax error: {exc}") from exc
    except SqlglotError as exc:
        raise SecurityValidationError(f"SQL parsing failure: {exc}") from exc

    # Filter out empty or None expressions
    valid_statements = [s for s in parsed_statements if s is not None]

    # Guardrail A: Exactly 1 statement (prohibits query chaining via semicolons)
    if len(valid_statements) == 0:
        raise SecurityValidationError("No valid SQL statement found.")
    if len(valid_statements) > 1:
        raise SecurityValidationError(
            f"Multiple SQL statements detected ({len(valid_statements)}). "
            f"Query chaining via semicolons is strictly prohibited."
        )

    root_stmt = valid_statements[0]

    # Guardrail B: Root expression must be exp.Select
    if not isinstance(root_stmt, exp.Select):
        root_type = type(root_stmt).__name__
        raise SecurityValidationError(
            f"Security policy violation: Root expression must be a SELECT query, "
            f"got {root_type}."
        )

    # Guardrail C: Explicitly check for forbidden destructive DDL/DML nodes in AST
    for forbidden_type in FORBIDDEN_EXPRESSION_TYPES:
        violating_nodes = list(root_stmt.find_all(forbidden_type))
        if violating_nodes:
            type_name = forbidden_type.__name__
            raise SecurityValidationError(
                f"Security policy violation: Prohibited destructive DDL/DML operation detected ({type_name})."
            )

    # Keyword safety scan on normalized representation to guard against unparsed dialect extensions
    sql_canonical = root_stmt.sql(dialect=dialect)
    for kw_pattern in BLOCKED_KEYWORDS:
        if re.search(kw_pattern, sql_canonical, re.IGNORECASE):
            match_word = re.search(kw_pattern, sql_canonical, re.IGNORECASE).group(0) # type: ignore
            raise SecurityValidationError(
                f"Security policy violation: Prohibited keyword '{match_word}' detected."
            )

    return sql_canonical


# Alias for backward compatibility
validate_and_sanitize_sql = validate_sql
