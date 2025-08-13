import random

# ----------------------------- Utilities -----------------------------

def jittered_backoff(attempt: int, base: float, max_delay: float) -> float:
    exp = min(max_delay, base * (2 ** attempt))
    # full jitter
    return random.uniform(0, exp)

