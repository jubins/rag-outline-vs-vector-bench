"""Simulated step times.

The bench does not call real models. Each model-like step sleeps for a time drawn from a
log-normal distribution around a median. Change the medians to match your own system.
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field


@dataclass
class LatencyModel:
    medians_ms: dict[str, float] = field(
        default_factory=lambda: {
            "embed": 40.0,
            "rerank": 80.0,
            "select": 800.0,
            "generate": 600.0,
        }
    )
    sigma: float = 0.25
    scale: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def sample_ms(self, step: str, factor: float = 1.0) -> float:
        median = self.medians_ms.get(step, 0.0)
        if median <= 0:
            return 0.0
        return median * factor * math.exp(self._rng.gauss(0.0, self.sigma))

    async def pause(self, step: str, factor: float = 1.0) -> None:
        """Sleep for a simulated step time. factor scales the median, for example with k."""
        delay = self.sample_ms(step, factor) * self.scale / 1000.0
        if delay > 0:
            await asyncio.sleep(delay)
