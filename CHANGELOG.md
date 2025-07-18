# Changelog

All notable changes to slicesim are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0]

The first release: a seeded, slot-level simulator of a 5G NR downlink cell
shared by eMBB, URLLC and mMTC slices, with classic schedulers and slicing
policies, and a PPO slice controller compared against them.

### Added

- **Scenarios** as YAML files validated with pydantic. Unknown keys are
  rejected, and every problem is reported at once as a path into the file.
- **Carrier**: numerology, slot duration and PRB count from TS 38.211 and
  TS 38.101-1 Table 5.3.2-1 for 15, 30 and 60 kHz subcarrier spacing.
- **Engine**: a slot loop with named, seeded random streams. Changing one
  model, or one UE group, never shifts another's random numbers.
- **Traffic**: full buffer, Poisson, periodic and on/off video-like models,
  with exact arrival times.
- **Radio channel**:
  - TR 38.901 urban macro path loss (LOS and NLOS), LOS probability and
    log-normal shadowing;
  - Rayleigh block fading with a configurable coherence time and diversity
    order.
- **Link adaptation**:
  - CQI per TS 38.214 Table 5.2.2.1-2 at 10% BLER, reported with a delay;
  - bits per PRB with the TS 38.306 overhead;
  - a BLER model;
  - HARQ with chase combining, a retransmission limit and in-order delivery.
- **Schedulers**: round robin, max C/I and proportional fair, compared by
  `slicesim compare`.
- **Network slicing**:
  - eMBB, URLLC and mMTC slices, each with an SLA and its own scheduler;
  - static, delay-aware and demand-proportional inter-slice policies;
  - SLA accounting per slice: violation ratio, satisfaction, delay
    percentiles, connection density;
  - low, medium and high load example scenarios.
- **Reinforcement learning** (the `rl` extra):
  - a Gymnasium environment, `slicesim/Slicing-v0`, that passes Gymnasium's
    environment checker;
  - PPO written from scratch in PyTorch;
  - `slicesim train` and `slicesim evaluate`;
  - a trained agent in `models/ppo-slicing.pt`.
- **Experiments**:
  - `slicesim experiment` sweeps load levels, policies and seeds into CSV
    files;
  - `slicesim plot` draws the figures (the `plots` extra);
  - results and figures in `docs/`.
- **Dashboard**: a Streamlit app that builds and runs a sliced scenario (the
  `dashboard` extra).

[0.1.0]: https://github.com/raoalqama-dev/slicesim/releases/tag/v0.1.0
