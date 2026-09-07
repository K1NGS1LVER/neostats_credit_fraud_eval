"""
Rate Limiter for Talk-to-Data Engine.

Provides a thread-safe Token-Bucket rate limiter tracking:
- Requests-Per-Minute (RPM) [default: 30]
- Tokens-Per-Minute (TPM) [default: 60,000]

Includes exponential backoff with random jitter for HTTP 429 errors
and inspection of available capacity/headroom.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimitExceededError(Exception):
    """Raised when rate limit capacity is exceeded and wait timeout expires."""
    pass


class TokenBucketRateLimiter:
    """
    Thread-safe dual Token-Bucket rate limiter tracking both
    Requests-Per-Minute (RPM) and Tokens-Per-Minute (TPM).
    """

    def __init__(
        self,
        max_rpm: int = 30,
        max_tpm: int = 60_000,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Initialize the rate limiter.

        Args:
            max_rpm: Maximum allowable requests per minute (default 30).
            max_tpm: Maximum allowable tokens per minute (default 60,000).
            clock_fn: Monotonic clock provider for time tracking.
        """
        if max_rpm <= 0 or max_tpm <= 0:
            raise ValueError("max_rpm and max_tpm must be positive integers.")

        self.max_rpm = float(max_rpm)
        self.max_tpm = float(max_tpm)
        self.rpm_fill_rate = self.max_rpm / 60.0  # requests per second
        self.tpm_fill_rate = self.max_tpm / 60.0  # tokens per second

        self.clock_fn = clock_fn
        self._lock = threading.Lock()

        # Start buckets fully replenished
        self._rpm_tokens = self.max_rpm
        self._tpm_tokens = self.max_tpm
        self._last_refill_time = self.clock_fn()

    def _refill_unlocked(self, now: float) -> None:
        """Replenish tokens proportional to elapsed time (caller must hold lock)."""
        elapsed = max(0.0, now - self._last_refill_time)
        if elapsed > 0:
            self._rpm_tokens = min(self.max_rpm, self._rpm_tokens + elapsed * self.rpm_fill_rate)
            self._tpm_tokens = min(self.max_tpm, self._tpm_tokens + elapsed * self.tpm_fill_rate)
            self._last_refill_time = now

    def acquire(
        self,
        estimated_tokens: int = 100,
        wait: bool = True,
        timeout: Optional[float] = 15.0,
    ) -> bool:
        """
        Acquire 1 request unit and estimated token units from the bucket.

        Args:
            estimated_tokens: Estimated token count for this request (default: 100).
            wait: If True, blocks until enough capacity is available or timeout expires.
            timeout: Maximum seconds to wait before raising RateLimitExceededError.

        Returns:
            True if tokens were acquired.

        Raises:
            RateLimitExceededError: If capacity cannot be acquired within timeout.
        """
        tokens_needed = max(1.0, float(estimated_tokens))
        start_time = self.clock_fn()

        while True:
            with self._lock:
                now = self.clock_fn()
                self._refill_unlocked(now)

                has_rpm = self._rpm_tokens >= 1.0
                has_tpm = self._tpm_tokens >= tokens_needed

                if has_rpm and has_tpm:
                    self._rpm_tokens -= 1.0
                    self._tpm_tokens -= tokens_needed
                    return True

                if not wait:
                    raise RateLimitExceededError(
                        f"Rate limit exceeded: RPM remaining={self._rpm_tokens:.1f}/{self.max_rpm:.0f}, "
                        f"TPM remaining={self._tpm_tokens:.0f}/{self.max_tpm:.0f}"
                    )

                # Calculate required sleep duration to satisfy deficit
                rpm_deficit = max(0.0, 1.0 - self._rpm_tokens)
                tpm_deficit = max(0.0, tokens_needed - self._tpm_tokens)

                time_for_rpm = rpm_deficit / self.rpm_fill_rate if self.rpm_fill_rate > 0 else 0.0
                time_for_tpm = tpm_deficit / self.tpm_fill_rate if self.tpm_fill_rate > 0 else 0.0
                sleep_needed = max(time_for_rpm, time_for_tpm)

            # Check timeout bounds
            elapsed = self.clock_fn() - start_time
            if timeout is not None:
                remaining_time = timeout - elapsed
                if sleep_needed > remaining_time:
                    raise RateLimitExceededError(
                        f"Rate limit exceeded: Required wait {sleep_needed:.2f}s exceeds timeout {timeout:.2f}s."
                    )

            # Sleep briefly outside the lock
            sleep_duration = min(sleep_needed, 0.25)
            time.sleep(max(0.01, sleep_duration))

    def get_headroom(self) -> Dict[str, Any]:
        """
        Inspect the current rate limit headroom and capacity.

        Returns:
            Dictionary with current RPM/TPM available, limits, and utilization percentages.
        """
        with self._lock:
            now = self.clock_fn()
            self._refill_unlocked(now)

            rpm_avail = round(self._rpm_tokens, 2)
            tpm_avail = round(self._tpm_tokens, 1)

            rpm_util = round(max(0.0, (1.0 - (rpm_avail / self.max_rpm)) * 100.0), 2)
            tpm_util = round(max(0.0, (1.0 - (tpm_avail / self.max_tpm)) * 100.0), 2)

            return {
                "rpm_limit": int(self.max_rpm),
                "rpm_available": rpm_avail,
                "rpm_utilization_pct": rpm_util,
                "tpm_limit": int(self.max_tpm),
                "tpm_available": tpm_avail,
                "tpm_utilization_pct": tpm_util,
                "has_capacity": (rpm_avail >= 1.0 and tpm_avail >= 1.0),
            }

    def reset(self) -> None:
        """Reset rate limiter buckets to maximum capacity."""
        with self._lock:
            self._rpm_tokens = self.max_rpm
            self._tpm_tokens = self.max_tpm
            self._last_refill_time = self.clock_fn()


def is_rate_limit_error(exception: Exception) -> bool:
    """Check whether an exception represents an HTTP 429 / Rate Limit error."""
    # Check status code attribute
    status_code = getattr(exception, "status_code", None)
    if status_code == 429:
        return True

    # Check response status code if present
    response = getattr(exception, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True

    # Check exception type name and string representation
    err_str = str(exception).lower()
    err_cls = exception.__class__.__name__.lower()

    if "ratelimit" in err_cls or "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str:
        return True

    return False


def extract_retry_after(exception: Exception) -> Optional[float]:
    """Extract retry-after delay in seconds from exception headers or message if available."""
    # Check response headers
    response = getattr(exception, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {})
        if "retry-after" in headers:
            try:
                return float(headers["retry-after"])
            except (ValueError, TypeError):
                pass

    # Check regex for 'try again in X.Xs' or 'retry after X'
    err_str = str(exception)
    match = re.search(r"try again in\s+([0-9.]+)\s*s", err_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            pass

    return None


def execute_with_backoff(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    **kwargs: Any,
) -> T:
    """
    Execute a function with exponential backoff and random jitter upon HTTP 429 rate limit errors.

    Args:
        func: The callable to execute.
        *args: Positional arguments for func.
        max_retries: Maximum retry attempts on 429 (default 3).
        initial_delay: Initial sleep duration in seconds.
        max_delay: Maximum sleep duration in seconds.
        backoff_factor: Multiplier for exponential backoff.
        jitter: If True, adds random jitter to prevent thundering herd.
        **kwargs: Keyword arguments for func.

    Returns:
        The return value of func.

    Raises:
        Exception: The last caught exception if retries are exhausted or non-429 error occurs.
    """
    attempt = 0
    current_delay = initial_delay

    while True:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if not is_rate_limit_error(exc) or attempt >= max_retries:
                raise

            attempt += 1

            # Check if API provided an explicit retry-after hint
            explicit_wait = extract_retry_after(exc)
            if explicit_wait is not None:
                delay = min(max_delay, explicit_wait + (0.2 if jitter else 0.0))
            else:
                delay = min(max_delay, current_delay * (backoff_factor ** (attempt - 1)))
                if jitter:
                    delay = delay * random.uniform(0.75, 1.25)

            logger.warning(
                "HTTP 429 Rate Limit encountered. Retrying attempt %d/%d after %.2fs backoff. Error: %s",
                attempt,
                max_retries,
                delay,
                exc,
            )
            time.sleep(delay)
            current_delay = delay
