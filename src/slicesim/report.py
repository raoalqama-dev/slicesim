"""Plain-text tables for the command line."""

from __future__ import annotations

from collections.abc import Sequence

from slicesim.config import Scenario
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


def render_summary(summary: RunSummary, scenario: Scenario) -> str:
    """The summary ``slicesim run`` prints."""
    cell = summary.cell
    title = summary.scenario
    if scenario.description:
        title += f": {scenario.description}"
    header = [
        title,
        f"  seed      {summary.seed}",
        f"  duration  {cell.duration_s:g} s ({cell.n_slots:,} slots)",
        f"  carrier   {summary.carrier}",
        f"  link      fixed, {scenario.link.bits_per_prb} bits per PRB "
        f"(cell capacity {cell.capacity_mbps:,.1f} Mbit/s)",
        f"  cell      {cell.delivered_mbps:,.2f} Mbit/s delivered, "
        f"{cell.prb_utilization:.1%} of PRBs used",
    ]
    headers = [
        "group",
        "UEs",
        "traffic",
        "offered Mb/s",
        "delivered Mb/s",
        "per UE Mb/s",
        "packets",
        "delay mean ms",
        "p50",
        "p95",
        "p99",
        "queued kB",
    ]
    rows = []
    for g in summary.groups:
        packets = MISSING if g.offered_mbps is None else f"{g.packets_delivered:,}"
        delay = g.delay
        rows.append(
            [
                g.name,
                f"{g.ues:,}",
                g.traffic,
                _number(g.offered_mbps),
                _number(g.delivered_mbps),
                _number(g.per_ue_mbps, 3),
                packets,
                _number(None if delay is None else delay.mean_ms),
                _number(None if delay is None else delay.p50_ms),
                _number(None if delay is None else delay.p95_ms),
                _number(None if delay is None else delay.p99_ms),
                _number(None if g.offered_mbps is None else g.backlog_bytes / 1e3, 1),
            ]
        )
    return "\n".join([*header, "", format_table(headers, rows, "lrl" + "r" * 9)])
