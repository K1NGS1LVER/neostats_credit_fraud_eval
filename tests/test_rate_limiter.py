import pytest
import time
from src.talk_to_data.rate_limiter import TokenBucketRateLimiter, execute_with_backoff, RateLimitExceededError

def test_rate_limiter_acquisition():
    limiter = TokenBucketRateLimiter(max_rpm=60, max_tpm=1000)
    acquired = limiter.acquire(estimated_tokens=10, timeout=1.0)
    assert acquired is True
    headroom = limiter.get_headroom()
    assert headroom['rpm_available'] < 60
    assert headroom['tpm_available'] <= 990

def test_rate_limiter_headroom_structure():
    limiter = TokenBucketRateLimiter(max_rpm=30, max_tpm=60000)
    headroom = limiter.get_headroom()
    assert "rpm_limit" in headroom
    assert "tpm_limit" in headroom
    assert "rpm_available" in headroom
    assert "tpm_available" in headroom
    assert headroom["rpm_limit"] == 30

def test_rate_limiter_retry_helper():
    call_count = 0
    def mock_api_call():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise Exception("429 Too Many Requests")
        return "SUCCESS"
    
    res = execute_with_backoff(mock_api_call, max_retries=2, initial_delay=0.01)
    assert res == "SUCCESS"
    assert call_count == 2
