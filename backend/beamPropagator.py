# beamPropagator.py
from dataclasses import dataclass
from typing import Generator, List, Any
import numpy as np


@dataclass
class PropagationCheckpoint:
    """Data at a single s-position during propagation."""
    s: float
    particles: np.ndarray
    element_index: int
    element: Any
    is_element_boundary: bool


def propagate(beamline: List,
              particles: np.ndarray,
              interval: float,
              rounding: int = 4) -> Generator[PropagationCheckpoint, None, None]:
    """
    Propagate particles through beamline, yielding at each checkpoint.

    Samples particle state at regular intervals and element boundaries.
    Initial state (s=0) is always included.
    """
    EPS = 1e-12
    s = 0.0
    current = particles.copy()

    yield PropagationCheckpoint(
        s=0.0,
        particles=current.copy(),
        element_index=-1,
        element=None,
        is_element_boundary=True
    )

    for idx, segment in enumerate(beamline):
        remaining = segment.length

        while remaining - interval > EPS:
            current = np.array(segment.useMatrice(current, length=interval))
            s = round(s + interval, rounding)
            remaining -= interval

            yield PropagationCheckpoint(
                s=s,
                particles=current.copy(),
                element_index=idx,
                element=segment,
                is_element_boundary=False
            )

        if remaining > EPS:
            current = np.array(segment.useMatrice(current, length=remaining))
            s = round(s + remaining, rounding)

            yield PropagationCheckpoint(
                s=s,
                particles=current.copy(),
                element_index=idx,
                element=segment,
                is_element_boundary=True
            )