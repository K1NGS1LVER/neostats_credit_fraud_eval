import pytest
from src.talk_to_data.validator import validate_sql, SecurityValidationError

def test_valid_select_queries():
    valid_queries = [
        "SELECT * FROM loan_applications LIMIT 10;",
        "SELECT NAME_EDUCATION_TYPE, AVG(TARGET) FROM loan_applications GROUP BY NAME_EDUCATION_TYPE",
        "SELECT COUNT(*), AVG(AMT_CREDIT) FROM loan_applications WHERE TARGET = 1",
        "SELECT OCCUPATION_TYPE, AVG(AMT_CREDIT) AS avg_credit FROM loan_applications GROUP BY OCCUPATION_TYPE ORDER BY avg_credit DESC LIMIT 5",
        "SELECT (SUM(CASE WHEN TARGET = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) AS default_rate FROM loan_applications"
    ]
    for q in valid_queries:
        sanitized = validate_sql(q)
        assert sanitized.upper().startswith("SELECT")

def test_blocked_dml_ddl_queries():
    malicious_queries = [
        "DROP TABLE loan_applications;",
        "DELETE FROM loan_applications WHERE TARGET = 1;",
        "UPDATE loan_applications SET TARGET = 0;",
        "INSERT INTO loan_applications VALUES (1, 2, 3);",
        "ALTER TABLE loan_applications DROP COLUMN TARGET;",
        "TRUNCATE TABLE loan_applications;",
        "CREATE TABLE test (id INT);"
    ]
    for q in malicious_queries:
        with pytest.raises(SecurityValidationError):
            validate_sql(q)

def test_blocked_stacked_queries():
    stacked = "SELECT * FROM loan_applications; DROP TABLE loan_applications;"
    with pytest.raises(SecurityValidationError):
        validate_sql(stacked)
