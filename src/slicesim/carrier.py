"""NR carrier numerology and transmission bandwidth (FR1).

The subcarrier spacing fixes the numerology mu, and with it the slot duration
(TS 38.211 section 4.2 and 4.3.2: 15 * 2**mu kHz, 2**mu slots per 1 ms
subframe, 14 OFDM symbols per slot with normal cyclic prefix). The channel
bandwidth and subcarrier spacing together fix the number of resource blocks,
from the maximum transmission bandwidth configuration N_RB in TS 38.101-1
Table 5.3.2-1.
"""

from __future__ import annotations

from dataclasses import dataclass

SUBCARRIERS_PER_PRB = 12
"""Subcarriers in one resource block (TS 38.211 section 4.4.4.1)."""

SYMBOLS_PER_SLOT = 14
"""OFDM symbols per slot with normal cyclic prefix (TS 38.211 Table 4.3.2-1)."""

# TS 38.101-1 Table 5.3.2-1: maximum transmission bandwidth configuration
# N_RB, keyed by subcarrier spacing in kHz, then channel bandwidth in MHz.
# Combinations the table marks N/A are left out.
NRB_TABLE: dict[int, dict[int, int]] = {
    15: {5: 25, 10: 52, 15: 79, 20: 106, 25: 133, 30: 160, 35: 188, 40: 216, 45: 242, 50: 270},
    30: {
        5: 11, 10: 24, 15: 38, 20: 51, 25: 65, 30: 78, 35: 92, 40: 106, 45: 119, 50: 133,
        60: 162, 70: 189, 80: 217, 90: 245, 100: 273,
    },
    60: {
        10: 11, 15: 18, 20: 24, 25: 31, 30: 38, 35: 44, 40: 51, 45: 58, 50: 65,
        60: 79, 70: 93, 80: 107, 90: 121, 100: 135,
    },
}  # fmt: skip


def supported_bandwidths(scs_khz: int) -> list[int]:
    """Channel bandwidths in MHz that TS 38.101-1 defines for a subcarrier spacing."""
    if scs_khz not in NRB_TABLE:
        raise ValueError(
            f"subcarrier spacing {scs_khz} kHz is not supported in FR1; "
            f"choose one of {', '.join(str(s) for s in NRB_TABLE)} kHz"
        )
    return list(NRB_TABLE[scs_khz])


@dataclass(frozen=True)
class Carrier:
    """A downlink NR carrier: its numerology, slot timing and resource blocks."""

    bandwidth_mhz: int
    scs_khz: int
    numerology: int
    n_prb: int

    @classmethod
    def create(cls, bandwidth_mhz: int, scs_khz: int) -> Carrier:
        """Look up a carrier, rejecting combinations TS 38.101-1 does not define."""
        bandwidths = supported_bandwidths(scs_khz)
        if bandwidth_mhz not in NRB_TABLE[scs_khz]:
            raise ValueError(
                f"a {bandwidth_mhz} MHz channel is not defined at {scs_khz} kHz subcarrier "
                f"spacing (TS 38.101-1 Table 5.3.2-1); choose one of "
                f"{', '.join(str(b) for b in bandwidths)} MHz"
            )
        numerology = (scs_khz // 15).bit_length() - 1
        return cls(
            bandwidth_mhz=bandwidth_mhz,
            scs_khz=scs_khz,
            numerology=numerology,
            n_prb=NRB_TABLE[scs_khz][bandwidth_mhz],
        )

    @property
    def slots_per_subframe(self) -> int:
        """Slots in each 1 ms subframe, 2**mu."""
        return 1 << self.numerology

    @property
    def slot_duration_s(self) -> float:
        """Slot duration in seconds: 1 ms at 15 kHz, 0.5 ms at 30 kHz, 0.25 ms at 60 kHz."""
        return 1e-3 / self.slots_per_subframe

    @property
    def prb_bandwidth_hz(self) -> float:
        """Bandwidth of one resource block, 12 subcarriers."""
        return SUBCARRIERS_PER_PRB * self.scs_khz * 1e3

    @property
    def resource_elements_per_prb(self) -> int:
        """Resource elements in one resource block over one slot, before overhead."""
        return SUBCARRIERS_PER_PRB * SYMBOLS_PER_SLOT

    def __str__(self) -> str:
        return (
            f"{self.bandwidth_mhz} MHz at {self.scs_khz} kHz (numerology {self.numerology}): "
            f"{self.n_prb} PRBs, {self.slot_duration_s * 1e3:g} ms slots"
        )
