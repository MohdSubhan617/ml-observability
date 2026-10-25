"""Shared statistical utilities for reproducible evaluation scripts."""

import math

WILSON_Z_95 = 1.959963984540054


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = WILSON_Z_95,
) -> tuple[float, float] | None:
    """Return a Wilson score confidence interval for a binomial proportion."""
    if trials < 0:
        raise ValueError("trials must be >= 0.")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials.")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and > 0.")
    if trials == 0:
        return None

    proportion = successes / trials
    z_squared = z * z
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z_squared / (4 * trials * trials))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
