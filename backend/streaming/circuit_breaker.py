"""
backend/streaming/circuit_breaker.py
Adaptive rate limiters and circuit breakers for non-blocking asynchronous APIs.
Prevents API throttling, rate-limit bans, and cascading failures across feeds.
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Optional
from loguru import logger


class CircuitState(str, Enum):
    CLOSED = "CLOSED"      # Normal operation, traffic allowed
    OPEN = "OPEN"          # Tripped, traffic blocked immediately
    HALF_OPEN = "HALF_OPEN"  # Testing single probe request


class AdaptiveRateLimiter:
    """
    Token bucket rate limiter with adaptive backoff.
    """
    def __init__(self, rate_per_second: float = 5.0, burst: int = 10):
        self.rate = rate_per_second
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= tokens

    def backoff(self, factor: float = 0.5) -> None:
        """Dynamically reduce rate on 429/rate-limit responses."""
        self.rate = max(0.1, self.rate * factor)
        logger.warning(f"RateLimiter throttled: new rate {self.rate:.2f} req/s")

    def restore(self, boost: float = 1.1) -> None:
        """Gradually restore rate on sustained successful responses."""
        self.rate = min(20.0, self.rate * boost)


class CircuitBreaker:
    """
    Fail-fast circuit breaker pattern for external APIs.
    """
    def __init__(
        self,
        name: str,
        failure_threshold: int = 4,
        recovery_timeout: float = 30.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    async def can_execute(self) -> bool:
        async with self._lock:
            now = time.monotonic()
            if self.state == CircuitState.OPEN:
                if (
                    self.last_failure_time
                    and (now - self.last_failure_time) > self.recovery_timeout
                ):
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"[{self.name}] Circuit switching to HALF_OPEN probe")
                    return True
                return False
            return True

    async def record_success(self) -> None:
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                logger.info(f"[{self.name}] Circuit recovered, switching to CLOSED")
            self.state = CircuitState.CLOSED
            self.failure_count = 0

    async def record_failure(self) -> None:
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(
                    f"[{self.name}] Circuit TRIPPED to OPEN ({self.failure_count} errors). "
                    f"Blocked for {self.recovery_timeout}s"
                )

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        fallback: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Execute an async callable safely with circuit breaker protection."""
        if not await self.can_execute():
            logger.warning(f"[{self.name}] Circuit OPEN - returning fallback")
            return fallback

        try:
            result = await func(*args, **kwargs)
            await self.record_success()
            return result
        except Exception as e:
            logger.warning(f"[{self.name}] Call error: {e}")
            await self.record_failure()
            return fallback
