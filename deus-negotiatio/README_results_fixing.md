# Oxford–Hyde Park SensorFusionDQN – Debugging Notes

## 1. Connection and Environment Fixes

- Ensure SUMO runs in server mode with `--remote-port 65500` and that the TraCI client uses the same port.
- Start SUMO before the client (or use `traci.start([...,"--remote-port","65500", ...])` and do not run a separate SUMO instance).
- At the end of each episode, call `traci.close()` (and `sumo_process.wait()` if using `subprocess`) to avoid TCP shutdown errors.
- Set the `SUMO_HOME` environment variable properly and include `$SUMO_HOME/tools` in `PYTHONPATH` to remove XML validation warnings.

## 2. Scenario and Metrics Corrections

- Fix the test scenario so that demand is active for the entire 720 s episode; current tests sometimes show `Total Vehicles Processed` equal to 0 or 1.
- Use the same network, routes, and demand that were used during training (LOS C 2046 projected layout).
- Define baseline vs AI comparison explicitly:
  - Compute `delay_baseline` and `delay_ai` on identical demand.
  - Define delay savings as `delay_baseline − delay_ai` (positive = AI better).
  - Log average congestion, average delay, throughput, and delay savings per episode.
- Add early termination for gridlock (e.g. avg delay > 60 s for 120 consecutive steps) with a large negative episode reward.

## 3. Reward and State Redesign

- Redefine the reward to directly penalize queues and delay:
  ```python
  reward_t = - (α * total_queue + β * total_delay)
  # Optional: add a small phase-change penalty to discourage rapid switching
  ```
- Verify that the observation/state fed to SensorFusionDQN at evaluation time is identical to training:
  - Same lane ordering and sensor channels.
  - Same normalization / scaling.
  - State dimension remains 172 with no missing or extra features.
- If the network or sensors changed since training, retrain the model or update the observation mapping accordingly.

## 4. Controlled Evaluation Protocol

- Build a standard evaluation suite:
  - Fixed-time (or actuated) baseline controller.
  - SensorFusionDQN controller with `epsilon = 0` in eval mode.
- For each scenario (seed), run:
  - Baseline episode: record avg delay, avg queue, throughput.
  - AI episode: same demand and seed.
- Compute and report:
  - `Δdelay = delay_baseline − delay_ai`
  - `Δqueue = queue_baseline − queue_ai`
- Aggregate results over multiple episodes (e.g. 10–20) to reduce variance.

## 5. Training Pipeline Checks

- Replay several episodes with both random and trained policies and log:
  - Chosen phase, queues, delays, reward, and "AI improvement" metric per step.
- If high rewards coexist with negative delay savings, adjust the reward function so that maximizing reward also minimizes delay.
- Confirm that evaluation loads the correct checkpoint (`bestmodel.pth` corresponding to the best reward) and that exploration is disabled in eval.

## 6. Next Steps

1. Stabilize TraCI connection and SUMO environment.
2. Fix the test demand so vehicles actually traverse the network.
3. Align reward with delay/queue minimization.
4. Re-run controlled experiments vs baseline and update this file with new plots and statistics.
5. Once delay savings become consistently positive, freeze the environment and training configuration and tag a stable version in the repository.
