# xplane-pilot

Milestone 1 is implemented as a deterministic fixed-wing pilot core with:

- typed flight state, guidance targets, and actuator commands
- a mode manager for takeoff through rollout
- direct-to navigation plus 45-to-downwind pattern geometry
- TECS-lite longitudinal control and bank/pitch PID inner loops
- a small deterministic simulator and unit tests for the milestone scenario

Run the test suite with:

```bash
uv run python -m unittest discover -s tests -v
```

Run the demo scenario with:

```bash
uv run python -m sim_pilot
```
