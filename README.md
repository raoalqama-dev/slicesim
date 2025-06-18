# slicesim

Discrete-time 5G RAN slicing simulator comparing classic schedulers with a
reinforcement-learning slice controller.

slicesim simulates a 5G NR downlink cell, slot by slot, shared by three network
slices (eMBB, URLLC and mMTC), and asks whether a PPO agent can meet per-slice
SLAs better than classic MAC schedulers and fixed slicing policies. Every run is
seeded, scenarios are validated YAML files, and the radio models cite the 3GPP
specifications they implement.

**Status:** early development. The simulation core works: a carrier sized from
TS 38.101-1, a seeded slot loop, four traffic models, per-packet delay
measurement and validated scenario files. Every resource block still carries a
fixed number of bits; the radio channel and link adaptation are next on the
[roadmap](#roadmap).

## Quick start

Python 3.11 or newer:

```sh
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e .
slicesim run scenarios/basic.yaml
```

```text
basic: four traffic types sharing a 20 MHz cell
  seed      1
  duration  10 s (20,000 slots)
  carrier   20 MHz at 30 kHz (numerology 1): 51 PRBs, 0.5 ms slots
  link      fixed, 500 bits per PRB (cell capacity 51.0 Mbit/s)
  cell      50.67 Mbit/s delivered, 100.0% of PRBs used

group    UEs  traffic      offered Mb/s  delivered Mb/s  per UE Mb/s  packets  delay mean ms   p50    p95    p99  queued kB
-------  ---  -----------  ------------  --------------  -----------  -------  -------------  ----  -----  -----  ---------
video      8  on_off              15.75           15.73        1.967    1,572           8.80  8.80  11.43  11.94       19.8
control   10  periodic             0.80            0.80        0.080    9,998           0.78  0.79   0.87   0.87        0.2
sensors   50  poisson              0.16            0.16        0.003      971           0.74  0.74   0.97   0.99        0.0
bulk       2  full_buffer             -           33.98       16.992        -              -     -      -      -          -
```

Each row is one UE group: the load it offered and the throughput it got, the
packets delivered to it, their delay from arrival at the gNB to delivery, and
what was still queued at the end. Full-buffer UEs always have data waiting, so
they have no offered load, packets or delay; here two of them soak up the
capacity the others leave.

`--seed N` and `--duration SECONDS` override the scenario, and `--json` prints
the same summary as JSON. The same scenario and seed always print the same
result.

## Scenarios

A scenario is a YAML file describing one run completely:

```yaml
name: basic
description: four traffic types sharing a 20 MHz cell   # optional
seed: 1
duration_s: 10

carrier:
  bandwidth_mhz: 20   # with the subcarrier spacing, must be in TS 38.101-1 Table 5.3.2-1
  scs_khz: 30         # 15, 30 or 60

link:                 # optional; this is the default
  model: fixed
  bits_per_prb: 500

ues:
  - name: video
    count: 8
    traffic:
      model: on_off
      mean_on_ms: 2000
      mean_off_ms: 1000
      frame_rate_fps: 30
      frame_bytes: 12500
```

The traffic models:

| `model` | Parameters | Behaviour |
| ------- | ---------- | --------- |
| `full_buffer` | none | always more data waiting than the cell can send |
| `poisson` | `rate_pps`, `packet_bytes` | fixed-size packets arriving as a Poisson process |
| `periodic` | `period_ms`, `packet_bytes` | one packet per period, each UE at a random phase (URLLC style) |
| `on_off` | `mean_on_ms`, `mean_off_ms`, `frame_rate_fps`, `frame_bytes` | video-like: frames at a fixed rate during exponentially distributed on periods |

Files are checked before anything runs. Unknown keys, impossible values and
carrier combinations the 3GPP table does not define are rejected, and every
problem is reported at once:

```text
$ slicesim run broken.yaml
slicesim: broken.yaml: invalid scenario
  carrier: a 100 MHz channel is not defined at 15 kHz subcarrier spacing (TS 38.101-1 Table 5.3.2-1); choose one of 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 MHz
  ues[0].traffic.rate_pps: Input should be greater than 0 (got -5)
  ues[1].traffic.jitter_ms: unknown field
```

## How it works

**Carrier.** The subcarrier spacing sets the numerology mu and the slot
duration, 1 ms / 2^mu (TS 38.211 section 4.3.2): 1 ms at 15 kHz, 0.5 ms at
30 kHz, 0.25 ms at 60 kHz. The number of resource blocks comes from the maximum
transmission bandwidth table, TS 38.101-1 Table 5.3.2-1: 20 MHz at 30 kHz is 51
PRBs, 100 MHz at 30 kHz is 273.

**Slot loop.** Each slot the scheduler sees every UE's queue as it stands at the
start of the slot, the cell sends what it allocates, and the packets arriving
during the slot join the queues, to be sent from the next slot on. Arrival times
are exact rather than rounded to slots, and a packet's delay runs from its
arrival to the end of the slot that carries its last byte, so a packet split
across several slots counts as delivered in the last one. Per-UE quantities are
numpy arrays; 100 UEs for 10,000 slots take about three seconds on a laptop.

**Allocation.** Until the MAC schedulers arrive (feature 3), the PRBs are split
equally among the UEs with data waiting, no UE getting more than it can use;
when fewer blocks than UEs remain, the leftover blocks go round in turn.

**Randomness.** Every source of randomness has its own
`numpy.random.Generator`, derived from the scenario seed and the stream's name
(each UE group's traffic is its own stream, named after the group), never from
the order streams are created in. Changing one group's traffic model, or adding
a group, leaves every other group's arrivals exactly as they were.

**Traffic.** Poisson arrivals draw a Poisson count per slot and spread the
arrivals uniformly within it. Periodic arrival k is computed as
`phase + k * period`, so long runs do not drift. On/off sources start in their
stationary state, and the first frame of each on period falls at a random point
in the first frame interval. Pinning it to the start of the period would add
half a frame to every on period, overshooting the configured load by 10% when on
periods last five frames on average.

## Roadmap

- [x] Simulation core: carrier tables, seeded slot engine, traffic models, metrics, YAML scenarios
- [ ] Radio channel and link adaptation (TR 38.901 UMa, CQI per TS 38.214, HARQ)
- [ ] MAC schedulers: round robin, max C/I, proportional fair; `slicesim compare`
- [ ] Network slicing with per-slice SLAs and inter-slice policies
- [ ] Gymnasium environment for the slice controller
- [ ] PPO agent written from scratch; `slicesim train` and `slicesim evaluate`
- [ ] Experiments, figures, dashboard and v0.1.0

## Development

```sh
pip install -e ".[dev]"
ruff check . && ruff format --check .   # lint and formatting
mypy                                    # strict type checking of src and tests
pytest --cov                            # tests, with a 90% coverage floor
```

CI runs the same checks, the tests on Python 3.11 to 3.14, and the tests again
against the oldest dependency versions `pyproject.toml` allows.
