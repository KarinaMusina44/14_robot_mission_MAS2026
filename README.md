# Self-Organization of Robots in a Hostile Environment

**Group:** 14  
**Date:** 16 March 2026  
**Members:** Deodato V. Bastos Neto, Karina Musina

---

## 1. Overview and Objectives

This project is an Agent-Based Model (ABM) of a distributed multi-agent system where autonomous robots cooperate to clean a hostile radioactive environment.

The grid is decomposed into three west-to-east zones: Z1 (low radioactivity), Z2 (medium), Z3 (high). Three robot types (Green, Yellow, Red) are zone-constrained and must collect, transform, and transport waste toward a final disposal zone.

**Primary objective:** clear all waste as fast as possible.

Our primary metric is `time_to_clear` per run (first step where system waste reaches zero), and the analysis is centered on reducing this value.

**Secondary preferences:**

- Avoid saturating communication channels when possible.
- Prefer configurations with fewer robots at equal `time_to_clear` (lower budget needs), even if we consider budget extensible overall.

**Tracked but not primary metric:**

- `cumulative_moves` (overall and per color) is recorded, and can be relevant as a proxy for travel energy/fuel, but it is not the primary KPI and is not the focus of the final analysis.

---

## 2. Requirements and Execution

### Prerequisites

Install Python dependencies:

```bash
pip install mesa solara matplotlib pandas
```

### A. Visual Mode (Solara UI)

```bash
solara run server.py
```

### B. Single Run CLI (`run.py`)

```bash
python run.py --steps 100 --n-waste 30 --verbose
```

`run.py` is intended for one simulation run and basic diagnostics. It currently exposes options such as:

- Grid and counts: `--steps`, `--width`, `--height`, `--n-robots`, `--n-green-robots`, `--n-yellow-robots`, `--n-red-robots`, `--n-waste`, `--n-green-wastes`
- Coordination/logging: `--green-coordination` / `--no-green-coordination`, `--log-communications` / `--no-log-communications`
- Utility: `--seed`, `--verbose`, `--report-every`, `--check-only`, `--debug-traceback`, `--model-class`

### C. Batch Experiments (`batch_experiments.py` / `run_batch.sh`)

Run the full experiment suite:

```bash
./run_batch.sh
```

Or run the batch runner directly:

```bash
python batch_experiments.py --help
```

`batch_experiments.py` is the script that exposes experimental toggles such as `--use-communication`, `--use-memory`, `--patrol-border`, `--multiple-wastes`, and `--vision` sweeps.


---

## 3. Theoretical Framework and M&S Scope

Following Modeling and Simulation (M&S) framing:

- **Source System:** hazardous waste handling in a radiation-constrained facility.
- **Experimental Frame:** evaluate how behavior toggles affect cleanup performance.
- **Model:** `RobotMissionModel` manages spatial dynamics, movement constraints, and action effects.
- **Simulator:** Mesa discrete-time stepping.

---

## 4. System Properties (Lecture 1)

In strict accordance with the foundational properties of Multi-Agent Systems:

### Environment Properties

- **Partially Observable:** Robots do not possess a global map. They rely on local views, perceiving only their current cell and a limited radius of visible tiles.
- **Dynamic:** The environment changes due to processes beyond a single agent's control as other agents continuously modify the distribution of waste.
- **Discrete:** Space is a 2D grid; time advances in discrete step intervals.
- **Decentralized:** There is no global master, planner, or "God-view" orchestrating the cleanup.

### Agent Properties

- **Autonomy:** Each robot operates independently, maintaining its own internal `knowledge` base and deciding its next action via a strict procedural loop: *percepts -> deliberate -> do*.
- **Loosely Coupled:** Agents do not share memory or access each other's code. Interaction is limited to physical environment modification (stigmergy) or explicit, limited message passing.

---

## 5. Interaction, Communication & Trade-offs (Lecture 3)

In a hostile, radioactive environment, communication is limited and must stay lightweight.  
Each agent has an `inbox` and exchanges short messages to keep the pipeline flowing without central control.

### Protocol Overview
- **`dropped_waste` (Green/Yellow/Red):** shares drop coordinates so downstream agents can intercept waste quickly.
- **`holding_one` (Green and Yellow):** deadlock negotiation when an agent holds exactly one unit; the lower id yields by dropping.
  Here, a deadlock means agents are blocked in a waiting loop (for example, several agents each carry one unit and all wait for a second unit before transforming), so no one progresses.
- **`green_visible_targets` (Green only):** shared-target arbitration to avoid multiple green robots chasing the same waste.
- **`disposal_zone` (Red):** one-time broadcast of discovered disposal location to other red robots.

Frustration mechanism (fallback): when communication/negotiation does not unblock the situation, an agent that has been holding one unit for too long drops it, so the pipeline can move again instead of staying stuck.

### Trade-off
Communication reduces `time_to_clear` and run-to-run variance by reducing random wandering and deadlocks, but it increases message overhead.  
We balance this with:
1. **Memory pruning:** stale coordinates are removed from `known_wastes` (and from `yielded_wastes` for Green/Yellow).
2. **Throttled deadlock signals:** `holding_one` is broadcast every 10 steps (not every step).
3. **Targeted sharing:** `disposal_zone` is broadcast once, and green arbitration is limited to visible peers.

---

## 6. Behavioral Strategies (Conceptual Configurations)

These are conceptual parameter settings used in analysis:

### Strategy 1: Reactive Random Walk

- **Configuration:** `use_communication=False`, `vision=1`, `patrol_border=False`
- **Mechanism:** randomized exploration + frustration timeout (if an agent keeps exactly one unit for too long, it drops it to break waiting loops).
- **Expected effect:** slower cleanup and higher variability in harder settings.

### Strategy 2: Higher-Vision Local Pathfinding (No Communication)

- **Configuration:** `use_communication=False`, higher `vision` (for example 3)
- **Mechanism:** better local look-ahead and target interception.
- **Expected effect:** partial compensation for missing communication.

### Strategy 3: Cooperative Pipeline

- **Configuration:** `use_communication=True`, `green_coordination=True`, `use_memory=True`, `patrol_border=True`
- **Mechanism:** local coordination, disposal-memory reuse, and proactive handoff positioning.
- **Expected effect:** often lower time-to-clear.

---

## 7. Visualization Snapshot

![Simulation Interface](simulation.png)

*(Fig 1: Example run in the Solara interface.)*

---

## 8. Batch Experiments and Interpretation

The experiment suite is executed by `run_batch.sh`, which calls `batch_experiments.py` with explicit parameter ranges.

### Methodology Clarifications

- Batch outputs are written per experiment directory.
- Each configuration is repeated on 20 seeds (`0` to `19`) before aggregation.
- We mainly use a one-factor-at-a-time logic: one parameter varies while others are held fixed to isolate its effect.
- Exception: Experiment 8 is an intentional stress test where `n-green-robots` and `n-yellow-robots` vary together.
- `time_to_clear_all_waste` is extracted per run from the batch trajectories.
- Plot uncertainty conventions:
- Line plots (quantitative sweeps): mean with **95% confidence interval**.
- Boolean bar plots: mean with **standard deviation** error bars.

### 1) Impact of Communication

- **Run config:** `--use-communication True,False`
- **Question:** does explicit messaging improve throughput?
- **Interpretation:** in these runs, enabling communication is associated with lower mean time-to-clear.

![Experience 1](batch_results/exp_communication/plot_time_to_clear_vs_use_communication.png)

As illustrated in the results, enabling peer-to-peer communication improves overall system performance.

- Reduced Average Time: when communication is disabled (`False`), the system relies on random exploration and frustration timeouts, giving an average clearance time around 240 steps. Enabling communication (`True`) lowers this average to just under 200 steps.
- Increased Consistency: the error bars also show lower variability with communication, meaning runs are not only faster but more stable when agents can negotiate deadlocks and broadcast dropped-waste locations.

### 2) Green-to-Green Coordination

- **Run config:** `--green-coordination-values True,False`
- **Question:** does same-color arbitration reduce target conflicts?
- **Interpretation:** effect size on mean can be small depending on the setting; variance changes are often more visible.

![Experience 2](batch_results/exp_green_coordination/plot_time_to_clear_vs_green_coordination.png)

This experiment complements global communication tests by isolating same-color coordination in zone `z1`.

- With `green_coordination=True`, visible green robots avoid pursuing the same waste target at the same time.
- In this setup, the mean improvement is limited, but variance is smaller, which still indicates a coordination benefit in consistency.

### 3) Red Agent Memory

- **Run config:** `--use-memory True,False`
- **Question:** does remembering disposal location reduce search overhead?
- **Interpretation:** memory-enabled settings are associated with lower mean completion time in the shown runs.

![Experience 3](batch_results/exp_memory/plot_time_to_clear_vs_use_memory.png)

As in the communication experiment, enabling memory improves outcomes, with an even larger effect size.

- Strong Reduction in Average Time: without memory (`False`), red agents must rediscover the disposal area repeatedly, and average completion time rises above 500 steps. With memory (`True`), the average drops near 200 steps.
- Large Consistency Gain: variance is also much lower with memory enabled, indicating more predictable end-to-end behavior.

### 4) Border Patrol

- **Run config:** `--patrol-border True,False`
- **Question:** does proactive border waiting improve handoffs?
- **Interpretation:** this comparison can show slower or more variable outcomes with patrol enabled; congestion/interference is a plausible hypothesis, not a proven causal mechanism.

![Experience 4](batch_results/exp_patrol/plot_time_to_clear_vs_patrol_border.png)

Contrary to initial expectations, proactive border patrolling can reduce overall efficiency in this configuration.

- Slight Increase in Average Time: without patrol (`False`), average time is around 210 steps; with patrol (`True`), it increases to around 240 steps.
- Increased Variance: the spread also increases with patrol, indicating less predictable runs.

One plausible interpretation is that forcing Yellow and Red robots to wait near borders can create local congestion or reduce opportunistic pickups deeper in their zones.

### 5) Initial Waste Distribution

- **Run config:** `--multiple-wastes True,False`
- **Question:** does starting with mixed waste types alter pipeline dynamics?
- **Interpretation:** mixed starts are often harder and can increase time and spread in this setup.

![Experience 5](batch_results/exp_multiple_wastes/plot_time_to_clear_vs_multiple_wastes.png)

As expected, enabling mixed initial waste types (`multiple_wastes=True`) increases both average completion time and variability.

- Increase in Average Time: homogeneous initialization (`False`) is faster on average (around 190 steps), while mixed initialization (`True`) is slower (around 300 steps).
- Increase in Variance: mixed starts also produce wider error bars, showing less consistent trajectories.

This is consistent with a harder cold-start pipeline, where all agent types must coordinate immediately in a more complex initial state.

### 6) Quantitative Scaling (OFAT)

- **Run config (from `run_batch.sh`):**
- `--design ofat`
- `--n-waste 4,16,24,32,40,48,56`
- `--n-green-robots 1,2,3,4,6,7,8`
- `--n-yellow-robots 1,2,3,4,5`
- `--n-red-robots 1,2,3,4,5`
- `--vision 1,2,3,4,5`

This is a one-factor-at-a-time design around a shared anchor chosen from middle values (for this range: waste 32, green 4, yellow 3, red 3, vision 3).

![Experience 6 Green](batch_results/exp_scaling/plot_time_to_clear_vs_green_agents.png)
![Experience 6 Yellow](batch_results/exp_scaling/plot_time_to_clear_vs_yellow_agents.png)
![Experience 6 Red](batch_results/exp_scaling/plot_time_to_clear_vs_red_agents.png)
![Experience 6 Vision](batch_results/exp_scaling/plot_time_to_clear_vs_vision.png)
![Experience 6 Waste](batch_results/exp_scaling/plot_time_to_clear_vs_waste.png)

Impact of Agent Population

- Green Agents: evaluated from `1 -> 8` robots (others fixed to OFAT anchor) to estimate early-stage throughput gains and potential diminishing returns.
- Yellow Agents: evaluated from `1 -> 5` robots to measure scaling of the intermediate transformation stage.
- Red Agents: evaluated from `1 -> 5` robots to analyze downstream disposal bottlenecks.

Sensory and Workload Scaling

- Vision Radius: evaluated from `1 -> 5` to estimate the effect of local perception on search/routing efficiency.
- Workload: evaluated from `4 -> 56` initial waste units to observe how completion time evolves with increasing load.

### 7) Vision as Communication Fallback

- **Run config:** `--vision 1,2,3,4,5 --use-communication False`
- **Question:** how much can perception compensate when messaging is disabled?
- **Interpretation:** higher vision is associated with faster completion in this scenario.

![Experience 7](batch_results/exp_vision_no_comm/plot_time_to_clear_vs_vision.png)

As expected, increasing visual range significantly improves performance when communication is disabled.

- Decreased Average Time: with `comm=False`, agents rely entirely on perception; moving from vision `1` to higher values reduces mean completion time strongly.
- Reduced Variance: the 95% CI narrows as vision increases, indicating more stable outcomes.

This supports the idea that stronger local perception can partially compensate for missing wireless coordination.

### 8) Extreme Crowding Stress Test

- **Run config:** `--n-green-robots 10,15 --n-yellow-robots 8,12 --n-waste 16`
- **Question:** behavior under high contention and low resources.
- **Interpretation:** results indicate non-uniform sensitivity across agent types under crowding.

![Experience 8 Green](batch_results/exp_extreme_crowding/plot_time_to_clear_vs_green_agents.png)
![Experience 8 Yellow](batch_results/exp_extreme_crowding/plot_time_to_clear_vs_yellow_agents.png)

Under high-density conditions, scaling effects differ by agent type.

- Green Scaling Saturation: increasing green robots from 10 to 15 has limited effect on mean time-to-clear, suggesting early-stage saturation and/or congestion.
- Yellow Bottleneck Sensitivity: increasing yellow robots from 8 to 12 shows a much stronger improvement, consistent with yellow agents being a central throughput bottleneck.
- Sequential Dependency: yellow throughput directly influences red-stage completion, so improvements in the middle stage propagate to end-to-end time.

### 9) Lone-Wolf Baseline

- **Run config:** `--n-green-robots 1 --n-yellow-robots 1 --n-red-robots 1 --n-waste 16,32,64`
- **Question:** baseline throughput without same-color peer interactions.
- **Interpretation:** useful reference curve for comparing multi-robot configurations.

![Experience 9](batch_results/exp_lone_wolf/plot_time_to_clear_vs_waste.png)

As expected, increasing initial waste raises mean time-to-clear in the lone-wolf configuration.

- Near-Linear Trend: from `16 -> 64` waste units, completion time rises in a broadly linear fashion.
- Baseline Robustness: this run family provides a useful reference curve for comparing multi-robot speedups and crowding effects.
