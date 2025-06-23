"""Plain-text tables for the command line."""

from __future__ import annotations

from collections.abc import Sequence

from slicesim.config import AdaptiveLink, FixedLink, Scenario
from slicesim.metrics import RunSummary

MISSING = "-"


def format_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], align: str | None = None
) -> str:
    """Align columns into a table.

    ``align`` has one character per column, ``l`` or ``r``; by default the
    first column is left-aligned and the rest right-aligned.
    """
    align = align or "l" + "r" * (len(headers) - 1)
    if len(align) != len(headers) or set(align) - {"l", "r"}:
        raise ValueError("align needs one 'l' or 'r' per column")
    widths = [len(h) for h in headers]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError("every row needs one cell per header")
        widths = [max(w, len(cell)) for w, cell in zip(widths, row, strict=True)]

    def line(cells: Sequence[str]) -> str:
        padded = [
            cell.ljust(w) if a == "l" else cell.rjust(w)
            for cell, w, a in zip(cells, widths, align, strict=True)
        ]
        return "  ".join(padded).rstrip()

    rule = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), rule, *(line(r) for r in rows)])


def _number(value: float | None, digits: int = 2) -> str:
    return MISSING if value is None else f"{value:,.{digits}f}"


def describe_link(scenario: Scenario) -> list[str]:
    """Header lines describing the link and, for the adaptive link, the cell."""
    link = scenario.link
    if isinstance(link, FixedLink):
        return [f"  link      fixed, {link.bits_per_prb} bits per PRB, no decoding errors"]
    cell = scenario.cell
    channel = scenario.channel
    fading = (
        f"Rayleigh fading ({channel.coherence_ms:g} ms coherence)"
        if channel.fading == "rayleigh"
        else "no fading"
    )
    shadowing = "shadowing" if channel.shadowing else "no shadowing"
    return [
        f"  cell      UMa at {cell.frequency_ghz:g} GHz, {cell.radius_m:g} m radius, "
        f"{cell.tx_power_dbm:g} dBm, {shadowing}, {fading}",
        f"  link      CQI-adaptive at 10% BLER, {link.overhead:.0%} overhead, CQI "
        f"{link.cqi_delay_slots} slots old, HARQ every {link.harq_rtt_slots} slots "
        f"up to {link.max_transmissions} transmissions",
    ]


def render_summary(summary: RunSummary, scenario: Scenario) -> str:
    """The summary ``slicesim run`` prints."""
    cell = summary.cell
    title = summary.scenario
    if scenario.description:
        title += f": {scenario.description}"
    result = (
        f"  result    {cell.delivered_mbps:,.2f} Mbit/s delivered of "
        f"{cell.peak_mbps:,.1f} peak, {cell.prb_utilization:.1%} of PRBs used"
    )
    if isinstance(scenario.link, AdaptiveLink) and cell.bler is not None:
        assert cell.residual_bler is not None
        result += f", BLER {cell.bler:.1%}, {cell.residual_bler:.2%} of blocks dropped"
    header = [
        title,
        f"  seed      {summary.seed}",
        f"  duration  {cell.duration_s:g} s ({cell.n_slots:,} slots)",
        f"  carrier   {summary.carrier}",
        *describe_link(scenario),
        result,
    ]
    headers = [
        "group",
        "UEs",
        "traffic",
        "offered",
        "delivered",
        "per UE",
        "SINR",
        "CQI",
        "packets",
        "lost",
        "delay",
        "p50",
        "p95",
        "p99",
        "queued",
    ]
    rows = []
    for g in summary.groups:
        packet_traffic = g.offered_mbps is not None
        delay = g.delay
        rows.append(
            [
                g.name,
                f"{g.ues:,}",
                g.traffic,
                _number(g.offered_mbps),
                _number(g.delivered_mbps),
                _number(g.per_ue_mbps, 3),
                _number(g.mean_sinr_db, 1),
                _number(g.mean_cqi, 1),
                f"{g.packets_delivered:,}" if packet_traffic else MISSING,
                f"{g.packets_lost:,}" if packet_traffic else MISSING,
                _number(None if delay is None else delay.mean_ms),
                _number(None if delay is None else delay.p50_ms),
                _number(None if delay is None else delay.p95_ms),
                _number(None if delay is None else delay.p99_ms),
                _number(g.backlog_bytes / 1e3 if packet_traffic else None, 1),
            ]
        )
    legend = (
        "Throughput in Mbit/s; SINR in dB before fading; packets delivered and lost; "
        "delay (mean and percentiles) in ms; queued in kB."
    )
    table = format_table(headers, rows, "lrl" + "r" * 12)
    return "\n".join([*header, "", table, "", legend])
