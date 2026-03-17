"""Numpy type aliases used in batch contracts.

Convention:
- **Vector** (1-D): shape ``(N,)`` — one value per pair in the batch.
- **Matrix** (2-D): shape ``(N, C)`` — e.g. indicator columns per pair.

All aliases resolve to ``npt.NDArray`` and carry no runtime cost.
"""

from typing import NewType

import numpy as np
import numpy.typing as npt


type FloatVector = npt.NDArray[np.float64]
type FloatMatrix = npt.NDArray[np.float64]
type BoolVector = npt.NDArray[np.bool_]
type IntVector = npt.NDArray[np.int64]

CandleInterval = NewType("CandleInterval", str)
VenueCode = NewType("VenueCode", str)
ProductCode = NewType("ProductCode", str)
