"""
LLM Rate Limiter

Provides client-side rate limiting for LLM API calls to prevent quota exhaustion
and ensure fair usage across the system.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60  # Max requests per minute
    requests_per_hour: int = 1000  # Max requests per hour
    tokens_per_minute: int = 100000  # Max tokens per minute
    tokens_per_hour: int = 1000000  # Max tokens per hour
    burst_limit: int = 10  # Max burst requests
    cooldown_seconds: float = 1.0  # Minimum time between requests


@dataclass
class RateLimitState:
    """Tracks current rate limit state."""
    minute_requests: int = 0
    hour_requests: int = 0
    minute_tokens: int = 0
    hour_tokens: int = 0
    minute_start: float = field(default_factory=time.time)
    hour_start: float = field(default_factory=time.time)
    last_request_time: float = 0.0
    burst_count: int = 0
    burst_window_start: float = field(default_factory=time.time)


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: float, limit_type: str):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit_type = limit_type


class LLMRateLimiter:
    """
    Client-side rate limiter for LLM API calls.
    
    Features:
    - Request rate limiting (per minute/hour)
    - Token rate limiting (per minute/hour)
    - Burst protection
    - Automatic cooldown
    - Per-provider tracking
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._states: Dict[str, RateLimitState] = {}
        self._lock = asyncio.Lock()
        self._global_state = RateLimitState()
    
    def _get_state(self, provider: str) -> RateLimitState:
        """Get or create rate limit state for a provider."""
        if provider not in self._states:
            self._states[provider] = RateLimitState()
        return self._states[provider]
    
    def _reset_windows(self, state: RateLimitState) -> None:
        """Reset rate limit windows if they have expired."""
        current_time = time.time()
        
        # Reset minute window
        if current_time - state.minute_start >= 60:
            state.minute_requests = 0
            state.minute_tokens = 0
            state.minute_start = current_time
        
        # Reset hour window
        if current_time - state.hour_start >= 3600:
            state.hour_requests = 0
            state.hour_tokens = 0
            state.hour_start = current_time
        
        # Reset burst window (5 second window)
        if current_time - state.burst_window_start >= 5:
            state.burst_count = 0
            state.burst_window_start = current_time
    
    async def acquire(self, provider: str = "default", estimated_tokens: int = 0) -> None:
        """
        Acquire permission to make an API call.
        
        Args:
            provider: The LLM provider name
            estimated_tokens: Estimated tokens for this request
            
        Raises:
            RateLimitExceededError: If rate limit would be exceeded
        """
        async with self._lock:
            state = self._get_state(provider)
            self._reset_windows(state)
            
            current_time = time.time()
            
            # Check cooldown
            time_since_last = current_time - state.last_request_time
            if time_since_last < self.config.cooldown_seconds:
                wait_time = self.config.cooldown_seconds - time_since_last
                await asyncio.sleep(wait_time)
                current_time = time.time()
            
            # Check burst limit
            if state.burst_count >= self.config.burst_limit:
                retry_after = 5 - (current_time - state.burst_window_start)
                if retry_after > 0:
                    raise RateLimitExceededError(
                        f"Burst limit exceeded for {provider}",
                        retry_after=retry_after,
                        limit_type="burst"
                    )
            
            # Check minute request limit
            if state.minute_requests >= self.config.requests_per_minute:
                retry_after = 60 - (current_time - state.minute_start)
                raise RateLimitExceededError(
                    f"Minute request limit exceeded for {provider}",
                    retry_after=retry_after,
                    limit_type="requests_per_minute"
                )
            
            # Check hour request limit
            if state.hour_requests >= self.config.requests_per_hour:
                retry_after = 3600 - (current_time - state.hour_start)
                raise RateLimitExceededError(
                    f"Hour request limit exceeded for {provider}",
                    retry_after=retry_after,
                    limit_type="requests_per_hour"
                )
            
            # Check minute token limit
            if estimated_tokens > 0:
                if state.minute_tokens + estimated_tokens > self.config.tokens_per_minute:
                    retry_after = 60 - (current_time - state.minute_start)
                    raise RateLimitExceededError(
                        f"Minute token limit exceeded for {provider}",
                        retry_after=retry_after,
                        limit_type="tokens_per_minute"
                    )
                
                # Check hour token limit
                if state.hour_tokens + estimated_tokens > self.config.tokens_per_hour:
                    retry_after = 3600 - (current_time - state.hour_start)
                    raise RateLimitExceededError(
                        f"Hour token limit exceeded for {provider}",
                        retry_after=retry_after,
                        limit_type="tokens_per_hour"
                    )
            
            # Update counters
            state.minute_requests += 1
            state.hour_requests += 1
            state.burst_count += 1
            state.last_request_time = current_time
            
            if estimated_tokens > 0:
                state.minute_tokens += estimated_tokens
                state.hour_tokens += estimated_tokens
            
            logger.debug(f"Rate limit acquired for {provider}: "
                        f"minute={state.minute_requests}/{self.config.requests_per_minute}, "
                        f"hour={state.hour_requests}/{self.config.requests_per_hour}")
    
    async def record_tokens(self, provider: str, actual_tokens: int, estimated_tokens: int = 0) -> None:
        """
        Record actual token usage after a request completes.
        
        Args:
            provider: The LLM provider name
            actual_tokens: Actual tokens used
            estimated_tokens: Previously estimated tokens (to adjust)
        """
        async with self._lock:
            state = self._get_state(provider)
            
            # Adjust token counts if actual differs from estimated
            adjustment = actual_tokens - estimated_tokens
            if adjustment != 0:
                state.minute_tokens += adjustment
                state.hour_tokens += adjustment
    
    def get_stats(self, provider: str = "default") -> Dict[str, Any]:
        """Get current rate limit statistics for a provider."""
        state = self._get_state(provider)
        self._reset_windows(state)
        
        current_time = time.time()
        
        return {
            "provider": provider,
            "minute_requests": state.minute_requests,
            "minute_requests_limit": self.config.requests_per_minute,
            "minute_requests_remaining": max(0, self.config.requests_per_minute - state.minute_requests),
            "minute_tokens": state.minute_tokens,
            "minute_tokens_limit": self.config.tokens_per_minute,
            "hour_requests": state.hour_requests,
            "hour_requests_limit": self.config.requests_per_hour,
            "hour_tokens": state.hour_tokens,
            "hour_tokens_limit": self.config.tokens_per_hour,
            "burst_count": state.burst_count,
            "burst_limit": self.config.burst_limit,
            "seconds_until_minute_reset": max(0, 60 - (current_time - state.minute_start)),
            "seconds_until_hour_reset": max(0, 3600 - (current_time - state.hour_start))
        }
    
    def reset(self, provider: Optional[str] = None) -> None:
        """Reset rate limit state for a provider or all providers."""
        if provider:
            if provider in self._states:
                self._states[provider] = RateLimitState()
        else:
            self._states.clear()
            self._global_state = RateLimitState()


# Global rate limiter instance
_rate_limiter: Optional[LLMRateLimiter] = None


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> LLMRateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = LLMRateLimiter(config)
    return _rate_limiter


def rate_limited(provider: str = "default", estimated_tokens: int = 0):
    """
    Decorator for rate-limiting async functions.
    
    Usage:
        @rate_limited(provider="openai", estimated_tokens=1000)
        async def call_llm():
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()
            await limiter.acquire(provider, estimated_tokens)
            return await func(*args, **kwargs)
        return wrapper
    return decorator
