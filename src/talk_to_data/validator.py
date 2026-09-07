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

# Adversarial prompt injection and instruction override patterns
ADVERSARIAL_INJECTION_PATTERNS = (
    r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous\s+|prior\s+|safety\s+|system\s+)?(?:instructions?|prompts?|rules?|directives?|guardrails?|filters?)\b",
    r"\b(?:jailbreak|dan\s+mode|developer\s+mode|unrestricted\s+mode)\b",
    r"\b(?:reveal|show|print|display|leak|output)\s+(?:the\s+)?(?:system\s+prompt|secret|api\s*key|password|token|confidential\s+data)\b",
    r"\b(?:confidential|sensitive|secret|proprietary)\s+(?:data|information|records|credentials)\b",
    r"\b(?:act\s+as|roleplay\s+as)\s+(?!a\s+(?:credit|risk|loan|financial|underwriting)\b)",
    r"\b(?:you\s+are\s+now|from\s+now\s+on\s+you\s+are)\b",
)

# Explicit out-of-scope off-topic patterns (creative, chit-chat, trivia, general programming)
OUT_OF_SCOPE_INTENT_PATTERNS = (
    r"\b(?:poem|poetry|rhyme|haiku|limerick|ballad|verse)\b",
    r"\b(?:joke|jokes|riddle|funny\s+story|pun|comedy)\b",
    r"\b(?:weather|forecast|temperature|rain|climate|humidity)\b",
    r"\b(?:capital\s+of|president\s+of|prime\s+minister|population\s+of)\b",
    r"\b(?:recipe|cooking|bake|cake|dinner|pasta|soup|cookie|pizza)\b",
    r"\b(?:movie|movies|film|actor|actress|song|lyrics|singer|album|band)\b",
    r"\b(?:write\s+(?:a\s+)?(?:python\s+script|javascript|html|css|c\+\+|bash\s+script))\b",
    r"\b(?:translate\s+(?:this|the\s+following)?\s+to|french|spanish|german|chinese|japanese)\b",
    r"\b(?:horoscope|zodiac|astrology)\b",
    r"\b(?:who\s+won\s+the|world\s+cup|super\s+bowl|olympics)\b",
    r"\b(?:tell\s+me\s+a\s+story|fairy\s+tale)\b",
)

# In-scope credit risk domain anchors
IN_SCOPE_DOMAIN_KEYWORDS = (
    r"\b(?:credit|loan|loans|debt|borrow|borrower|borrowers|lending|underwrit(?:ing|er)?)\b",
    r"\b(?:default|defaults|defaulted|defaulting|delinquen(?:t|cy|cies)|past\s+due|dpd)\b",
    r"\b(?:risk|tier|score|scores|rating|ratings|probability|pd|lgd|ead|expected\s+loss)\b",
    r"\b(?:applicant|applicants|application|applications|client|clients|customer|customers)\b",
    r"\b(?:income|salary|earnings|annuity|annuities|goods\s+price|amount|balance)\b",
    r"\b(?:education|degree|occupation|occupations|job|profession|employed|employment|tenure)\b",
    r"\b(?:gender|male|female|age|birth|housing|family|status|children|car|realty|real\s+estate)\b",
    r"\b(?:ext_source|external\s+source|credit\s+bureau|bureau|inquir(?:y|ies))\b",
    r"\b(?:social\s+circle|def_30|def_60|contract\s+type|cash\s+loan|revolving)\b",
    r"\b(?:dti|leverage|credit[- ]to[- ]income|payment[- ]rate)\b",
    r"\b(?:portfolio|distribution|segment|quartile|bracket|approved|approval|rejected|repaid)\b",
    r"\b(?:loan_applications|sk_id_curr|target|amt_credit|amt_income_total|amt_annuity)\b",
)

# General aggregate query patterns relevant to the dataset
GENERAL_PORTFOLIO_PATTERNS = (
    r"\b(?:how\s+many|total\s+count|count\s+all|number\s+of\s+records|how\s+many\s+records|how\s+many\s+rows|dataset|database|table)\b",
    r"\b(?:summary\s+statistics|describe\s+data|top\s+\d+|bottom\s+\d+|highest|lowest|average|median|mean|records|samples)\b",
)


def check_query_scope(question: str) -> Tuple[bool, Optional[str]]:
    """
    Evaluate whether an incoming user query is strictly within the credit risk,
    loan underwriting, and portfolio analytics domain.

    Args:
        question: The user's natural language input.

    Returns:
        Tuple of (is_in_scope, refusal_reason).
        If is_in_scope is True, refusal_reason is None.
    """
    clean_q = question.strip()
    if not clean_q:
        return False, "Query is empty."

    # 1. Adversarial prompt injection or instruction override scan
    for pattern in ADVERSARIAL_INJECTION_PATTERNS:
        if re.search(pattern, clean_q, re.IGNORECASE):
            return False, "Adversarial instruction or prompt override attempt detected."

    # 2. Explicit out-of-scope off-topic intent scan
    for pattern in OUT_OF_SCOPE_INTENT_PATTERNS:
        if re.search(pattern, clean_q, re.IGNORECASE):
            return False, "Inquiry is outside the credit risk, loan underwriting, and portfolio analytics domain."

    # 3. Domain anchor and aggregate query relevance check
    has_domain_anchor = any(re.search(p, clean_q, re.IGNORECASE) for p in IN_SCOPE_DOMAIN_KEYWORDS)
    has_portfolio_pattern = any(re.search(p, clean_q, re.IGNORECASE) for p in GENERAL_PORTFOLIO_PATTERNS)

    if not (has_domain_anchor or has_portfolio_pattern):
        return False, "Query lacks credit risk, borrower, or loan portfolio context."

    return True, None


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
