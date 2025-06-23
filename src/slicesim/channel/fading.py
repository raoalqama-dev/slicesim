"""Small-scale fading.

Rayleigh block fading: each UE's channel power gain |h|^2 is exponentially
distributed with unit mean, held for a coherence time and then redrawn
independently. Each UE's blocks start at its own random offset, so the
cell's UEs do not all change at once. The gain is flat across the carrier.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class BlockFading:
    """Rayleigh block fading for ``n_ues`` UEs, advanced one slot at a time."""

    def __init__(self, n_ues: int, coherence_slots: int, rng: np.random.Generator) -> None:
        if coherence_slots < 1:
            raise ValueError("coherence_slots must be at least 1")
        self.coherence_slots = coherence_slots
        self.rng = rng
        self.offset = rng.integers(0, coherence_slots, n_ues)
        self.gain = rng.exponential(1.0, n_ues)
        self._slot = 0

    def gain_db(self, slot: int) -> npt.NDArray[np.float64]:
        """Each UE's fading gain in dB during ``slot``; slots must be consecutive."""
        if slot != self._slot:
            raise ValueError(f"expected slot {self._slot}, got {slot}")
        if slot > 0:
            redraw = np.flatnonzero((slot + self.offset) % self.coherence_slots == 0)
            if redraw.size:
                self.gain[redraw] = self.rng.exponential(1.0, redraw.size)
        self._slot += 1
        result: npt.NDArray[np.float64] = 10 * np.log10(self.gain)
        return result


class NoFading:
    """A channel without small-scale fading: every gain is 0 dB."""

    def __init__(self, n_ues: int) -> None:
        self._zeros = np.zeros(n_ues)
        self._zeros.flags.writeable = False

    def gain_db(self, slot: int) -> npt.NDArray[np.float64]:
        return self._zeros
