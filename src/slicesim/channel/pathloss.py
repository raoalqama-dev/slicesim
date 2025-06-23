"""Urban macro (UMa) large-scale propagation from 3GPP TR 38.901.

Path loss follows Table 7.4.1-1, the LOS probability Table 7.4.2-1 and the
shadow fading standard deviations the same Table 7.4.1-1 gives (4 dB LOS,
6 dB NLOS). Distances are in metres and the carrier frequency in GHz, as in
the tables. The UMa model assumes a 25 m base station; UE heights up to 13 m
keep the effective environment height h_E at 1 m (Table 7.4.1-1, note 1),
which this module relies on.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

UMA_BS_HEIGHT_M = 25.0
"""Base station antenna height the UMa model is defined for."""

UMA_MIN_DISTANCE_M = 35.0
"""Minimum BS-UT ground distance for UMa (TR 38.901 Table 7.2-1)."""

UMA_MAX_UE_HEIGHT_M = 13.0
"""Below this UE height the effective environment height is exactly 1 m."""

UMA_LOS_SHADOWING_DB = 4.0
UMA_NLOS_SHADOWING_DB = 6.0

SPEED_OF_LIGHT = 3.0e8
"""TR 38.901 Table 7.4.1-1 note 1 uses c = 3.0e8 m/s for the breakpoint distance."""

_ENVIRONMENT_HEIGHT_M = 1.0

FloatArray = npt.NDArray[np.float64]


def distance_3d(d2d: npt.ArrayLike, h_bs: float, h_ut: float) -> FloatArray:
    """Straight-line distance between the antennas, from the ground distance."""
    return np.hypot(np.asarray(d2d, dtype=np.float64), h_bs - h_ut)


def breakpoint_distance(fc_ghz: float, h_bs: float, h_ut: float) -> float:
    """The UMa breakpoint distance d'_BP = 4 h'_BS h'_UT f_c / c (Table 7.4.1-1, note 1)."""
    h_bs_eff = h_bs - _ENVIRONMENT_HEIGHT_M
    h_ut_eff = h_ut - _ENVIRONMENT_HEIGHT_M
    return 4 * h_bs_eff * h_ut_eff * fc_ghz * 1e9 / SPEED_OF_LIGHT


def _check(fc_ghz: float, h_ut: float) -> None:
    if not 0.5 <= fc_ghz <= 100:
        raise ValueError(f"UMa is defined for 0.5 to 100 GHz, not {fc_ghz} GHz")
    if not 1.5 <= h_ut <= UMA_MAX_UE_HEIGHT_M:
        raise ValueError(f"UE height must be 1.5 to {UMA_MAX_UE_HEIGHT_M:g} m, not {h_ut} m")


def uma_los(
    d2d: npt.ArrayLike, fc_ghz: float, h_bs: float = UMA_BS_HEIGHT_M, h_ut: float = 1.5
) -> FloatArray:
    """UMa LOS path loss in dB (TR 38.901 Table 7.4.1-1).

    PL1 = 28.0 + 22 log10(d_3D) + 20 log10(f_c) up to the breakpoint, then
    PL2 = 28.0 + 40 log10(d_3D) + 20 log10(f_c) - 9 log10(d'_BP^2 + (h_BS - h_UT)^2).
    """
    _check(fc_ghz, h_ut)
    d2d = np.asarray(d2d, dtype=np.float64)
    d3d = distance_3d(d2d, h_bs, h_ut)
    d_bp = breakpoint_distance(fc_ghz, h_bs, h_ut)
    freq = 20 * np.log10(fc_ghz)
    pl1 = 28.0 + 22 * np.log10(d3d) + freq
    pl2 = 28.0 + 40 * np.log10(d3d) + freq - 9 * np.log10(d_bp**2 + (h_bs - h_ut) ** 2)
    result: FloatArray = np.where(d2d <= d_bp, pl1, pl2)
    return result


def uma_nlos(
    d2d: npt.ArrayLike, fc_ghz: float, h_bs: float = UMA_BS_HEIGHT_M, h_ut: float = 1.5
) -> FloatArray:
    """UMa NLOS path loss in dB (TR 38.901 Table 7.4.1-1).

    PL'_NLOS = 13.54 + 39.08 log10(d_3D) + 20 log10(f_c) - 0.6 (h_UT - 1.5),
    and PL_NLOS = max(PL_LOS, PL'_NLOS), so NLOS is never better than LOS.
    """
    d3d = distance_3d(d2d, h_bs, h_ut)
    nlos = 13.54 + 39.08 * np.log10(d3d) + 20 * np.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
    result: FloatArray = np.maximum(uma_los(d2d, fc_ghz, h_bs, h_ut), nlos)
    return result


def uma_los_probability(d2d: npt.ArrayLike, h_ut: float = 1.5) -> FloatArray:
    """Probability that a UE at ground distance ``d2d`` has line of sight (Table 7.4.2-1).

    1 within 18 m; beyond, [18/d + exp(-d/63) (1 - 18/d)] [1 + C'(h_UT) 5/4 (d/100)^3 exp(-d/150)],
    with C'(h_UT) = 0 up to 13 m and ((h_UT - 13) / 10)^1.5 above.
    """
    d = np.maximum(np.asarray(d2d, dtype=np.float64), 1e-9)
    c_prime = 0.0 if h_ut <= 13.0 else ((h_ut - 13.0) / 10.0) ** 1.5
    far = (18 / d + np.exp(-d / 63) * (1 - 18 / d)) * (
        1 + c_prime * 1.25 * (d / 100) ** 3 * np.exp(-d / 150)
    )
    result: FloatArray = np.where(d <= 18.0, 1.0, far)
    return result
