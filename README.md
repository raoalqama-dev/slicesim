# slicesim

Discrete-time 5G RAN slicing simulator comparing classic schedulers with a
reinforcement-learning slice controller.

slicesim simulates a 5G NR downlink cell, slot by slot, shared by three network
slices (eMBB, URLLC and mMTC), and asks whether a PPO agent can meet per-slice
SLAs better than classic MAC schedulers and fixed slicing policies. Every run is
seeded, scenarios are validated YAML files, and the radio models cite the 3GPP
specifications they implement.

**Status:** early development. A cell sized from TS 38.101-1 serves UEs dropped
over a TR 38.901 urban macro cell, with path loss, shadowing and Rayleigh block
fading. Transport blocks are sized from the CQI each UE reports (TS 38.214),
fail with a BLER that depends on the SINR they meet, and are retransmitted by
HARQ. Round robin, max C/I and proportional fair schedulers share the PRBs, and
`slicesim compare` runs a scenario under each of them. Four traffic models,
per-packet delay and loss, and validated scenario files complete the core.
Network slicing is next on the [roadmap](#roadmap).

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
  cell      UMa at 3.5 GHz, 300 m radius, 46 dBm, shadowing, Rayleigh fading (40 ms coherence)
  link      CQI-adaptive at 10% BLER, 14% overhead, CQI 2 slots old, HARQ every 8 slots up to 4 transmissions
  scheduler round robin
  result    46.46 Mbit/s delivered of 81.8 peak, 100.0% of PRBs used, BLER 2.6%, 0.19% of blocks dropped

group    UEs  traffic      offered  delivered  per UE   Jain  SINR   CQI  packets  lost  delay   p50    p95    p99  buffer
-------  ---  -----------  -------  ---------  ------  -----  ----  ----  -------  ----  -----  ----  -----  -----  ------
video      8  on_off         15.75      15.71   1.963  0.954  28.6  12.7    1,548    23   9.21  7.08  20.54  39.28     1.2
control   10  periodic        0.80       0.80   0.080  1.000  34.5  13.1    9,997     0   0.91  0.79   0.87   4.78     0.0
sensors   50  poisson         0.16       0.16   0.003  0.945  24.2  12.2      970     0   1.00  0.76   1.39   4.97     0.0
bulk       2  full_buffer        -      29.80  14.899  0.935  16.4  10.4        -     -      -     -      -      -       -

Throughput in Mbit/s; Jain's index over the group's UEs; SINR in dB before fading; packets delivered and lost; delay (mean and percentiles) in ms; buffer is the average queue per UE in kB.
```

Each row is one UE group: the load it offered and the throughput it got, how
evenly that throughput was spread over its UEs (Jain's index), its mean SINR and
reported CQI, the packets delivered and lost, their delay from arrival at the gNB
to delivery, and the average queue per UE.
Full-buffer UEs always have data waiting, so they have no offered load, packets
or delay; here two of them soak up the capacity the others leave. The control
group's 99th percentile, 4.78 ms, is one HARQ retransmission (8 slots of
0.5 ms) on top of the usual one-slot wait.

`--seed N` and `--duration SECONDS` override the scenario, and `--json` prints
the same summary as JSON. The same scenario and seed always print the same
result.

## Comparing schedulers

`slicesim compare` runs a scenario once per scheduler and prints one row each.
Every run keeps the seed, and since every random stream is named rather than
drawn in order, all schedulers see the same UE positions, channels and packet
arrivals:

```text
$ slicesim compare scenarios/fairness.yaml
fairness: twenty full-buffer UEs across a 500 m cell
  ...

scheduler                           delivered   Jain  edge UE  median UE  delay  p50  p95  p99  buffer  BLER
----------------------------------  ---------  -----  -------  ---------  -----  ---  ---  ---  ------  ----
round robin                             38.80  0.736    0.606      1.554      -    -    -    -     0.0  2.9%
max C/I                                 80.86  0.218    0.000      0.000      -    -    -    -     0.0  0.8%
proportional fair (100 ms average)      45.79  0.783    0.848      1.974      -    -    -    -     0.0  3.4%
```

`--schedulers rr,pf` picks which to run. Jain's index here is over every UE's
throughput, which is meaningful when the UEs have alike traffic, as these
full-buffer ones do; the edge UE is the 5th percentile of per-UE throughput.

What the table shows:

- **Max C/I doubles the cell throughput and starves the edge.** It always
  serves the best channels, so the far UEs get nothing at all.
- **Proportional fair beats round robin on both throughput and fairness.** RR
  gives every UE the same PRBs every slot, whatever its fading. PF gives each UE
  its turn when its channel is good relative to its own average, so UEs are
  served near their fading peaks. The gain is largest where rate grows fastest
  with SINR, at low SINR, so the edge UEs gain the most (+40% at the 5th
  percentile, against +27% at the median) and throughput ends up more even than
  under RR. With equal channels, where only fading tells the UEs apart, the two
  are equally fair: both within 0.002 of perfect, and which edges ahead depends
  on the seed.
- **With packet traffic PF also cuts delay.** A UE that has been idle has a
  low average, so its next packet goes out first. In `scenarios/basic.yaml`
  the 99th percentile delay over every packet is 8.6 ms under PF, 18.5 ms under
  RR and over a second under max C/I.

## Scenarios

A scenario is a YAML file describing one run completely. Everything but the
name, seed, duration, carrier and UEs has a default:

```yaml
name: basic
description: four traffic types sharing a 20 MHz cell   # optional
seed: 1
duration_s: 10

carrier:
  bandwidth_mhz: 20   # with the subcarrier spacing, must be in TS 38.101-1 Table 5.3.2-1
  scs_khz: 30         # 15, 30 or 60

cell:                 # the defaults
  radius_m: 300
  min_distance_m: 35        # TR 38.901 Table 7.2-1
  ue_height_m: 1.5          # up to 13 m
  frequency_ghz: 3.5
  tx_power_dbm: 46          # total, shared equally over the PRBs
  antenna_gain_db: 0        # gNB plus UE
  noise_figure_db: 7
  interference_margin_db: 3 # noise rise standing in for the other cells

channel:              # the defaults
  shadowing: true
  fading: rayleigh          # or none
  coherence_ms: 40

link:                 # the defaults
  model: adaptive
  overhead: 0.14            # TS 38.306 section 4.1.2, FR1 downlink
  cqi_delay_slots: 2
  harq_rtt_slots: 8
  max_transmissions: 4

scheduler:            # rr (the default), maxci, or pf
  model: pf
  time_constant_ms: 100     # pf only: the throughput average's time constant

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

`link: {model: fixed, bits_per_prb: 500}` replaces the radio channel with a
fixed number of bits per PRB and no decoding errors, which is useful for
checking schedulers and traffic in isolation.

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

**Path loss and shadowing.** UEs are dropped uniformly over the cell's area,
between the minimum distance and the radius, around a 25 m gNB. Each draws a
line-of-sight state from the UMa LOS probability (TR 38.901 Table 7.4.2-1) and
gets the UMa LOS or NLOS path loss of Table 7.4.1-1, with the breakpoint
distance of its note 1, plus log-normal shadowing of 4 dB (LOS) or 6 dB (NLOS).
At 3.5 GHz and a 3D distance of 100 m that is 82.9 dB LOS and 102.6 dB NLOS.
UEs do not move, so these stay fixed for the run.

**Fading and SINR.** Each UE's channel power gain is exponential with unit mean
(Rayleigh), held for the coherence time and then redrawn, each UE's blocks
starting at its own random offset. The default 40 ms is the coherence time of a
pedestrian at 3.5 GHz (a 9.7 Hz Doppler shift, 0.423 / f_D). The SINR is the
transmit power per PRB plus antenna gain, minus path loss and shadowing, plus
fading, over thermal noise in one PRB (-174 dBm/Hz) raised by the noise figure
and the interference margin. Fading is flat across the carrier.

**Link adaptation.** Each UE reports the highest CQI of TS 38.214 Table
5.2.2.1-2 that its SINR supports at 10% BLER, and the scheduler sizes the
UE's transport block from the report it received `cqi_delay_slots` ago: a PRB
carries 12 x 14 x efficiency x (1 - overhead) bits, from 22 bits at CQI 1 to
802 at CQI 15. 3GPP leaves the SINR each CQI needs to the implementation; the
thresholds used are the 10% BLER points of AWGN link-level curves common in
system-level simulators. A block fails with a probability that is 10% at its
CQI's threshold and falls logistically with the margin above it, about 3.4 dB
from 90% to 1%. When the channel changed since the report, the block meets a
SINR the CQI was not chosen for, which is where most failures come from.

**HARQ.** A failed block is sent again `harq_rtt_slots` later on the same PRBs
at the same CQI, ahead of any new data, and the UE combines every copy (chase
combining: the SINRs add up). After `max_transmissions` failures the block is
dropped and its packets are lost. Packets reach the UE in order, as PDCP
delivers them: a packet that decoded waits for an earlier block still being
retransmitted, so a retransmission delays everything queued behind it, which is
part of what stretches the video group's 99th percentile above.

**Slot loop.** Each slot, retransmissions go first, then the scheduler shares
the remaining PRBs among the UEs with data waiting, as their queues stand at the
start of the slot; packets arriving during the slot wait for the next one.
Arrival times are exact rather than rounded to slots, and a packet's delay runs
to the end of the slot in which it is released. Per-UE quantities are numpy
arrays; 100 UEs for 10,000 slots take under six seconds on a laptop.

**Schedulers.** Each slot the scheduler learns how many PRBs are left after
retransmissions, how many each UE could fill, and how many bits a PRB carries
for each UE at its reported CQI, and decides how many PRBs each UE gets (the
channel is flat, so which ones does not matter). None gives a UE more than it
can use or leaves a PRB idle while someone could use it, and ties go to UEs in
turn.

- **Round robin** splits the PRBs equally among the UEs with data waiting,
  sharing again whatever a UE cannot use; leftover blocks go round in turn.
- **Max C/I** serves UEs in order of bits per PRB, each as much as it needs.
- **Proportional fair** serves UEs in order of bits per PRB over their average
  throughput, an exponentially weighted moving average of the bits received
  per slot with a time constant of `time_constant_ms`.

A UE whose reported CQI is 0 is out of coverage and gets nothing.

**Randomness.** Every source of randomness has its own
`numpy.random.Generator`, derived from the scenario seed and the stream's name,
never from the order streams are created in. Each UE group has its own traffic,
placement, LOS, shadowing and fading streams, named after the group, and
decoding has one more. Changing one group's traffic, or adding a group, leaves
every other group's arrivals and channel exactly as they were, and changing the
traffic never moves a UE or changes its fading.

**Traffic.** Poisson arrivals draw a Poisson count per slot and spread the
arrivals uniformly within it. Periodic arrival k is computed as
`phase + k * period`, so long runs do not drift. On/off sources start in their
stationary state, and the first frame of each on period falls at a random point
in the first frame interval. Pinning it to the start of the period would add
half a frame to every on period, overshooting the configured load by 10% when on
periods last five frames on average.

## Roadmap

- [x] Simulation core: carrier tables, seeded slot engine, traffic models, metrics, YAML scenarios
- [x] Radio channel and link adaptation (TR 38.901 UMa, CQI per TS 38.214, HARQ)
- [x] MAC schedulers: round robin, max C/I, proportional fair; `slicesim compare`
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
