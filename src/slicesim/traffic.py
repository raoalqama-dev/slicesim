"""Downlink traffic models.

Each source drives every UE of one group. The engine asks it, slot by slot,
for the packets arriving at the gNB during ``[t0, t1)``; the intervals must
be contiguous and increasing. Arrival times are exact, not rounded to slots,
so packet delays are measured from the moment a packet arrives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from slicesim.config import (
    FullBufferTraffic,
    OnOffTraffic,
    PeriodicTraffic,
    PoissonTraffic,
    TrafficConfig,
)


@dataclass(frozen=True)
class Arrivals:
    """Packets arriving in one interval, ordered by UE and then by time."""

    ue: npt.NDArray[np.intp]
    """UE index within the group."""
    time: npt.NDArray[np.float64]
    size: npt.NDArray[np.int64]
    """Packet sizes in bytes."""

    @classmethod
    def empty(cls) -> Arrivals:
        return _NO_ARRIVALS

    def __len__(self) -> int:
        return int(self.ue.size)


_NO_ARRIVALS = Arrivals(
    ue=np.empty(0, dtype=np.intp),
    time=np.empty(0, dtype=np.float64),
    size=np.empty(0, dtype=np.int64),
)
for _array in (_NO_ARRIVALS.ue, _NO_ARRIVALS.time, _NO_ARRIVALS.size):
    _array.flags.writeable = False


def _offsets(counts: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """For packets grouped by UE, each packet's position within its UE's group."""
    starts = np.cumsum(counts) - counts
    return np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(starts, counts)


class TrafficSource(ABC):
    """Packet arrivals for ``n_ues`` UEs, drawn from one random generator."""

    full_buffer = False
    """Whether the UEs always have data waiting instead of receiving packets."""

    def __init__(self, n_ues: int, rng: np.random.Generator) -> None:
        if n_ues <= 0:
            raise ValueError("a traffic source needs at least one UE")
        self.n_ues = n_ues
        self.rng = rng

    @abstractmethod
    def arrivals(self, t0: float, t1: float) -> Arrivals:
        """Packets arriving during ``[t0, t1)``."""

    @property
    @abstractmethod
    def mean_rate_bps(self) -> float:
        """Long-run offered load per UE in bits per second (infinite for full buffer)."""


class FullBuffer(TrafficSource):
    """UEs that always have more data than the cell can send."""

    full_buffer = True

    def arrivals(self, t0: float, t1: float) -> Arrivals:
        return Arrivals.empty()

    @property
    def mean_rate_bps(self) -> float:
        return float("inf")


class Poisson(TrafficSource):
    """Fixed-size packets arriving as independent Poisson processes per UE.

    The number of arrivals in an interval is Poisson distributed, and given
    that number the arrival times are independent and uniform over it.
    """

    def __init__(
        self, n_ues: int, rng: np.random.Generator, rate_pps: float, packet_bytes: int
    ) -> None:
        super().__init__(n_ues, rng)
        if rate_pps <= 0 or packet_bytes <= 0:
            raise ValueError("rate_pps and packet_bytes must be positive")
        self.rate_pps = rate_pps
        self.packet_bytes = packet_bytes

    def arrivals(self, t0: float, t1: float) -> Arrivals:
        counts = self.rng.poisson(self.rate_pps * (t1 - t0), self.n_ues).astype(np.int64)
        total = int(counts.sum())
        if total == 0:
            return Arrivals.empty()
        ue = np.repeat(np.arange(self.n_ues, dtype=np.intp), counts)
        time = t0 + self.rng.random(total) * (t1 - t0)
        order = np.lexsort((time, ue))
        return Arrivals(
            ue=ue[order],
            time=time[order],
            size=np.full(total, self.packet_bytes, dtype=np.int64),
        )

    @property
    def mean_rate_bps(self) -> float:
        return self.rate_pps * self.packet_bytes * 8


class Periodic(TrafficSource):
    """A packet every ``period_s`` per UE, each UE at its own uniform random phase.

    Arrival k of a UE is at ``phase + k * period``, computed from k rather than
    accumulated, so long runs do not drift.
    """

    def __init__(
        self, n_ues: int, rng: np.random.Generator, period_s: float, packet_bytes: int
    ) -> None:
        super().__init__(n_ues, rng)
        if period_s <= 0 or packet_bytes <= 0:
            raise ValueError("period_s and packet_bytes must be positive")
        self.period_s = period_s
        self.packet_bytes = packet_bytes
        self.phase = rng.random(n_ues) * period_s
        self._next_k = np.zeros(n_ues, dtype=np.int64)

    def arrivals(self, t0: float, t1: float) -> Arrivals:
        # Arrivals strictly before t1 are those with k < ceil((t1 - phase) / period).
        end_k = np.ceil((t1 - self.phase) / self.period_s).astype(np.int64)
        counts = np.maximum(end_k - self._next_k, 0)
        total = int(counts.sum())
        if total == 0:
            return Arrivals.empty()
        ue = np.repeat(np.arange(self.n_ues, dtype=np.intp), counts)
        k = self._next_k[ue] + _offsets(counts)
        self._next_k += counts
        return Arrivals(
            ue=ue,
            time=self.phase[ue] + k * self.period_s,
            size=np.full(total, self.packet_bytes, dtype=np.int64),
        )

    @property
    def mean_rate_bps(self) -> float:
        return self.packet_bytes * 8 / self.period_s


class OnOff(TrafficSource):
    """Video-like traffic: frames at a fixed rate during on periods.

    On and off periods are exponentially distributed. During an on period a
    frame arrives every ``1 / frame_rate`` seconds, the first one at a uniform
    random point within the first frame interval. Pinning the first frame to
    the start of the period would add half a frame to every on period on
    average; with a random phase an on period of length L carries L * rate
    frames on average, so the long-run load is exactly ``mean_rate_bps``.
    Each UE starts in the stationary distribution: on with probability
    ``mean_on / (mean_on + mean_off)``, with an exponential residual period.
    """

    def __init__(
        self,
        n_ues: int,
        rng: np.random.Generator,
        mean_on_s: float,
        mean_off_s: float,
        frame_rate_fps: float,
        frame_bytes: int,
    ) -> None:
        super().__init__(n_ues, rng)
        if min(mean_on_s, mean_off_s, frame_rate_fps) <= 0 or frame_bytes <= 0:
            raise ValueError("on/off traffic parameters must be positive")
        self.mean_on_s = mean_on_s
        self.mean_off_s = mean_off_s
        self.frame_interval_s = 1.0 / frame_rate_fps
        self.frame_bytes = frame_bytes
        self.on = rng.random(n_ues) < self.on_probability
        self.state_end = rng.exponential(np.where(self.on, mean_on_s, mean_off_s))
        # A UE that is already on is part way through a frame interval.
        self.next_frame = rng.random(n_ues) * self.frame_interval_s

    @property
    def on_probability(self) -> float:
        return self.mean_on_s / (self.mean_on_s + self.mean_off_s)

    def arrivals(self, t0: float, t1: float) -> Arrivals:
        busy = np.flatnonzero((self.on & (self.next_frame < t1)) | (self.state_end < t1))
        if busy.size == 0:
            return Arrivals.empty()
        ues: list[int] = []
        times: list[float] = []
        for i in busy.tolist():
            on = bool(self.on[i])
            end = float(self.state_end[i])
            frame = float(self.next_frame[i])
            while True:
                if on:
                    limit = min(end, t1)
                    while frame < limit:
                        ues.append(i)
                        times.append(frame)
                        frame += self.frame_interval_s
                    if end >= t1:
                        break
                    on = False
                    end += float(self.rng.exponential(self.mean_off_s))
                else:
                    if end >= t1:
                        break
                    on = True
                    frame = end + float(self.rng.random()) * self.frame_interval_s
                    end += float(self.rng.exponential(self.mean_on_s))
            self.on[i] = on
            self.state_end[i] = end
            self.next_frame[i] = frame
        return Arrivals(
            ue=np.asarray(ues, dtype=np.intp),
            time=np.asarray(times, dtype=np.float64),
            size=np.full(len(ues), self.frame_bytes, dtype=np.int64),
        )

    @property
    def mean_rate_bps(self) -> float:
        return self.frame_bytes * 8 / self.frame_interval_s * self.on_probability


def build_source(config: TrafficConfig, n_ues: int, rng: np.random.Generator) -> TrafficSource:
    """Create the source a scenario's traffic section describes."""
    match config:
        case FullBufferTraffic():
            return FullBuffer(n_ues, rng)
        case PoissonTraffic():
            return Poisson(n_ues, rng, config.rate_pps, config.packet_bytes)
        case PeriodicTraffic():
            return Periodic(n_ues, rng, config.period_ms * 1e-3, config.packet_bytes)
        case OnOffTraffic():
            return OnOff(
                n_ues,
                rng,
                config.mean_on_ms * 1e-3,
                config.mean_off_ms * 1e-3,
                config.frame_rate_fps,
                config.frame_bytes,
            )
