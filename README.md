# slicesim

Discrete-time 5G RAN slicing simulator comparing classic schedulers with a
reinforcement-learning slice controller.

slicesim simulates a 5G NR downlink cell, slot by slot, shared by three network
slices (eMBB, URLLC and mMTC), and asks whether a PPO agent can meet per-slice
SLAs better than classic MAC schedulers and fixed slicing policies. Every run is
seeded, scenarios are validated YAML files, and the radio models cite the 3GPP
specifications they implement.

**Status:** early development.
