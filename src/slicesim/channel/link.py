"""Link adaptation: from SINR to CQI to bits per resource block, and decoding errors.

The CQI table is TS 38.214 Table 5.2.2.1-2 (4-bit CQI, up to 64QAM), whose
entries a UE reports so that the transport block error probability stays at
or below 10% (TS 38.214 section 5.2.2.1). 3GPP leaves the SINR at which each
CQI meets that target to the implementation; the thresholds below are the
10% BLER points of AWGN link-level curves widely used in system-level
simulation (for example the Vienna LTE simulators), which the same modulation
and code rates share between LTE and NR.

A resource block over one slot carries 12 subcarriers x 14 symbols x the CQI's
spectral efficiency x (1 - overhead) bits. The default overhead, 0.14, is the
FR1 downlink value TS 38.306 section 4.1.2 uses for peak rates (reference
signals and control).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from slicesim.carrier import SUBCARRIERS_PER_PRB, SYMBOLS_PER_SLOT


class CqiEntry(NamedTuple):
    modulation: str
    code_rate_x1024: int
    efficiency: float
    """Information bits per resource element."""


# TS 38.214 Table 5.2.2.1-2, CQI 1 to 15. CQI 0 means out of range.
CQI_TABLE: tuple[CqiEntry, ...] = (
    CqiEntry("QPSK", 78, 0.1523),
    CqiEntry("QPSK", 120, 0.2344),
    CqiEntry("QPSK", 193, 0.3770),
    CqiEntry("QPSK", 308, 0.6016),
    CqiEntry("QPSK", 449, 0.8770),
    CqiEntry("QPSK", 602, 1.1758),
    CqiEntry("16QAM", 378, 1.4766),
    CqiEntry("16QAM", 490, 1.9141),
    CqiEntry("16QAM", 616, 2.4063),
    CqiEntry("64QAM", 466, 2.7305),
    CqiEntry("64QAM", 567, 3.3223),
    CqiEntry("64QAM", 666, 3.9023),
    CqiEntry("64QAM", 772, 4.5234),
    CqiEntry("64QAM", 873, 5.1152),
    CqiEntry("64QAM", 948, 5.5547),
)

CQI_SINR_THRESHOLDS_DB = np.array(
    [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7]
)
"""Lowest SINR at which CQI 1 to 15 decodes with at most 10% BLER."""

BLER_TARGET = 0.1

BLER_SLOPE_PER_DB = 2.0
"""Steepness of the BLER curve: about 3.4 dB from 90% to 1% BLER."""

DEFAULT_OVERHEAD = 0.14

_EFFICIENCY = np.array([0.0, *(e.efficiency for e in CQI_TABLE)])
_THRESHOLD = np.array([np.inf, *CQI_SINR_THRESHOLDS_DB])


def cqi_from_sinr(sinr_db: npt.ArrayLike) -> npt.NDArray[np.intp]:
    """The highest CQI meeting the 10% BLER target at each SINR; 0 when none does."""
    result: npt.NDArray[np.intp] = np.searchsorted(
        CQI_SINR_THRESHOLDS_DB, np.asarray(sinr_db, dtype=np.float64), side="right"
    )
    return result


def efficiency(cqi: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Spectral efficiency in bits per resource element; 0 for CQI 0."""
    result: npt.NDArray[np.float64] = _EFFICIENCY[np.asarray(cqi)]
    return result


def bits_per_prb(cqi: npt.ArrayLike, overhead: float = DEFAULT_OVERHEAD) -> npt.NDArray[np.int64]:
    """Bits one resource block carries in one slot at each CQI."""
    if not 0 <= overhead < 1:
        raise ValueError("overhead must be in [0, 1)")
    res = SUBCARRIERS_PER_PRB * SYMBOLS_PER_SLOT * (1 - overhead)
    return np.floor(res * efficiency(cqi)).astype(np.int64)


def bler(sinr_db: npt.ArrayLike, cqi: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Probability that a transport block sent at ``cqi`` fails at ``sinr_db``.

    A logistic curve in the SINR margin over the CQI's threshold, equal to the
    10% target at zero margin: 1 / (1 + 9 exp(slope * margin)). CQI 0 always
    fails.
    """
    margin = np.asarray(sinr_db, dtype=np.float64) - _THRESHOLD[np.asarray(cqi)]
    odds = (1 - BLER_TARGET) / BLER_TARGET * np.exp(np.clip(BLER_SLOPE_PER_DB * margin, -50, 50))
    result: npt.NDArray[np.float64] = 1.0 / (1.0 + odds)
    return result


def combine_db(sinr_db: npt.ArrayLike) -> float:
    """Effective SINR after chase-combining transmissions: the linear sum, in dB."""
    linear = np.power(10.0, np.asarray(sinr_db, dtype=np.float64) / 10)
    return float(10 * np.log10(linear.sum()))
