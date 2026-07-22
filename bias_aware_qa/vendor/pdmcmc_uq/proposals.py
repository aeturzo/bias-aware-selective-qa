from __future__ import annotations

import math


DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "carbon_kg_per_kwh": (1.0, math.inf),
    "repairability_score": (0.0, 10.0),
    "durability_score": (0.0, 10.0),
    "recycled_content": (0.0, 1.0),
    "energy_efficiency": (0.0, 1.0),
}


def reflect_to_bounds(value: float, lower: float, upper: float) -> float:
    """Reflect a random-walk proposal into bounds without hard clipping."""

    x = float(value)
    if lower > upper:
        raise ValueError("lower must be <= upper.")
    if lower == upper:
        return lower
    if not math.isfinite(lower) and not math.isfinite(upper):
        return x
    if not math.isfinite(upper):
        return float(lower + (lower - x)) if x < lower else x
    if not math.isfinite(lower):
        return float(upper - (x - upper)) if x > upper else x
    width = upper - lower
    while x < lower or x > upper:
        if x < lower:
            x = lower + (lower - x)
        if x > upper:
            x = upper - (x - upper)
        if x < lower - 10 * width or x > upper + 10 * width:
            x = lower + ((x - lower) % (2 * width))
            if x > upper:
                x = upper - (x - upper)
    return float(x)
