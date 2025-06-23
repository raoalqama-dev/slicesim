"""The cell: where UEs are, what each one's channel looks like, and its SINR per slot.

UEs are dropped uniformly over an annulus around the gNB, between the minimum
distance and the cell radius. Each UE draws a LOS state from the TR 38.901
UMa LOS probability at its distance, and a log-normal shadowing term with the
standard deviation of that state; both stay fixed for the run, since UEs do
not move. Block fading then varies the channel from slot to slot.

The SINR of every resource block is the same for a UE (the fading is flat):

    SINR = P_tx - 10 log10(N_PRB) + G - PL - SF + fading - (N_0 + NF + I)

where the transmit power is shared equally over the PRBs, the noise is
-174 dBm/Hz over one PRB's bandwidth plus the noise figure, and I is the
interference margin, a noise rise standing in for the other cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from slicesim.carrier import Carrier
from slicesim.channel.fading import BlockFading, NoFading
from slicesim.channel.pathloss import (
    UMA_BS_HEIGHT_M,
    UMA_LOS_SHADOWING_DB,
    UMA_NLOS_SHADOWING_DB,
    distance_3d,
    uma_los,
    uma_los_probability,
    uma_nlos,
)
from slicesim.config import CellConfig, ChannelConfig

THERMAL_NOISE_DBM_PER_HZ = -174.0

FloatArray = npt.NDArray[np.float64]


def drop_ues(
    n: int, radius_m: float, min_distance_m: float, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """Uniform positions over the annulus ``min_distance_m <= r <= radius_m``.

    The radius is drawn as sqrt(U(r_min^2, r_max^2)), which makes the density
    uniform in area rather than in distance.
    """
    r = np.sqrt(rng.uniform(min_distance_m**2, radius_m**2, n))
    theta = rng.uniform(0.0, 2 * math.pi, n)
    return r * np.cos(theta), r * np.sin(theta)


def noise_per_prb_dbm(carrier: Carrier, noise_figure_db: float) -> float:
    """Thermal noise over one resource block, plus the receiver noise figure."""
    return THERMAL_NOISE_DBM_PER_HZ + 10 * math.log10(carrier.prb_bandwidth_hz) + noise_figure_db


@dataclass(frozen=True)
class GroupStreams:
    """The generators one UE group's channel draws from."""

    drop: np.random.Generator
    los: np.random.Generator
    shadowing: np.random.Generator
    fading: np.random.Generator


@dataclass
class GroupChannel:
    """The large-scale channel of one UE group, and its fading process."""

    x_m: FloatArray
    y_m: FloatArray
    los: npt.NDArray[np.bool_]
    pathloss_db: FloatArray
    shadowing_db: FloatArray
    fading: BlockFading | NoFading


class RadioChannel:
    """Every UE's channel, group by group, in the engine's UE order.

    Each group draws its positions, LOS states, shadowing and fading from its
    own generators, so changing one group never moves another's UEs or
    changes their channels.
    """

    def __init__(
        self,
        carrier: Carrier,
        cell: CellConfig,
        channel: ChannelConfig,
        groups: list[tuple[int, GroupStreams]],
    ) -> None:
        """``groups`` gives each group's UE count and random streams, in UE order."""
        self.carrier = carrier
        self.cell = cell
        coherence_slots = max(1, round(channel.coherence_ms * 1e-3 / carrier.slot_duration_s))
        self.coherence_slots = coherence_slots
        self.groups: list[GroupChannel] = []
        for count, rngs in groups:
            x, y = drop_ues(count, cell.radius_m, cell.min_distance_m, rngs.drop)
            d2d = np.hypot(x, y)
            los = rngs.los.random(count) < uma_los_probability(d2d, cell.ue_height_m)
            pathloss = np.where(
                los,
                uma_los(d2d, cell.frequency_ghz, UMA_BS_HEIGHT_M, cell.ue_height_m),
                uma_nlos(d2d, cell.frequency_ghz, UMA_BS_HEIGHT_M, cell.ue_height_m),
            )
            sigma = np.where(los, UMA_LOS_SHADOWING_DB, UMA_NLOS_SHADOWING_DB)
            shadowing = (
                rngs.shadowing.normal(0.0, 1.0, count) * sigma
                if channel.shadowing
                else np.zeros(count)
            )
            fading: BlockFading | NoFading = (
                BlockFading(count, coherence_slots, rngs.fading)
                if channel.fading == "rayleigh"
                else NoFading(count)
            )
            self.groups.append(GroupChannel(x, y, los, pathloss, shadowing, fading))

        self.x_m = np.concatenate([g.x_m for g in self.groups])
        self.y_m = np.concatenate([g.y_m for g in self.groups])
        self.los = np.concatenate([g.los for g in self.groups])
        self.pathloss_db = np.concatenate([g.pathloss_db for g in self.groups])
        self.shadowing_db = np.concatenate([g.shadowing_db for g in self.groups])
        rx_per_prb = (
            cell.tx_power_dbm
            - 10 * math.log10(carrier.n_prb)
            + cell.antenna_gain_db
            - self.pathloss_db
            - self.shadowing_db
        )
        noise = noise_per_prb_dbm(carrier, cell.noise_figure_db) + cell.interference_margin_db
        self.mean_sinr_db: FloatArray = rx_per_prb - noise
        """Each UE's SINR before fading: path loss, shadowing and the link budget."""

    @property
    def distance_m(self) -> FloatArray:
        """Ground distance from the gNB."""
        result: FloatArray = np.hypot(self.x_m, self.y_m)
        return result

    @property
    def distance_3d_m(self) -> FloatArray:
        return distance_3d(self.distance_m, UMA_BS_HEIGHT_M, self.cell.ue_height_m)

    def sinr_db(self, slot: int) -> FloatArray:
        """Every UE's SINR during ``slot``; slots must be consecutive."""
        fading = np.concatenate([g.fading.gain_db(slot) for g in self.groups])
        result: FloatArray = self.mean_sinr_db + fading
        return result
