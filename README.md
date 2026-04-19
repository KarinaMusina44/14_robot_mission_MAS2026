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

## 3. Theoretical Framework and M&S Scope (Lecture 2)

Following Modeling and Simulation (M&S) theory principles:

- **Source System:** a hazardous waste logistics process where tasks are sequential (collect, transform, handoff, dispose) and must be solved under radiation and zone constraints.
- **Experimental Frame:** evaluate how behavior toggles (communication, coordination, memory, patrol, sensing range, and robot counts) change cleanup speed.
- **Model:** `RobotMissionModel` represents the environment (`MultiGrid`), zone restrictions, action rules (`move`, `pickup`, `transform`, `drop`, `put_away`), and inter-agent message exchange.
- **Simulator:** Mesa executes the model in discrete time, with repeated runs under controlled parameter settings to compare strategies.


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
  Here, a deadlock means the mission cannot finish because agents are stuck : each one carries one unit, no free waste remains to pick up, and nobody can complete the next transformation step.
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

## 6. Behavioral Strategies & Configurations Tested

We implemented toggles to test different evolutionary building blocks of MAS behaviors.

### Strategy 1: Reactive Random Walk (No Communication)
* **Configuration:** `--use-communication False`, `--vision 1`, `--patrol-border False`
* **Mechanism:** Robots wander randomly. Deadlocks are solved via a **Frustration Timeout**: If an agent holds 1 waste for >20 steps without finding a second, it drops the waste out of "frustration".
* **Result:** Lowest communication overhead (0 messages), but exponentially higher `cumulative_moves` and `time_to_clear`.

### Strategy 2: Cognitive Smart Pathfinding (No Communication)
* **Configuration:** `--use-communication False`, `--vision 3`
* **Mechanism:** Agents use an extended vision radius to "look ahead" and use Manhattan-distance pathfinding to intercept wastes.
* **Result:** Proves that an efficient, localized movement mechanism can compensate for a lack of communication up to a certain grid density.

### Strategy 3: Fully Cooperative Network (Communication + Memory + Patrol)
* **Configuration:** `--use-communication True`, `--green-coordination`, `--use-memory True`, `--patrol-border True`
* **Mechanism:** Green agents use local coordination (`green_visible_targets` + `holding_one`) to reduce target conflicts and deadlocks. Red agents memorize the disposal zone coordinates upon discovery. Yellow/Red agents proactively navigate to and patrol the zone borders when their inventory is empty, waiting for handoffs.
* **Result:** Highest communication overhead, but incredibly fast `time_to_clear`.

---

## 7. Experimental Results & Visualization

Below is a snapshot of the Solara interface during an active simulation run. The custom UI dynamically renders continuous radiation zones, robot inventories (diamonds), and the disposal zone (star).

![Simulation Interface](simulation.png)
Fig 1: Simulation running at Step 35. Green agents in Z1, Yellow in Z2, Red in Z3 exploring towards the disposal star.

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

- Reduced Average Time: when communication is disabled (`False`), the system relies on random exploration and frustration timeouts, giving an average clearance time around 280 steps. Enabling communication (`True`) lowers this average to just under 250 steps.
- Increased Consistency: the error bars also show lower variability with communication, meaning runs are not only faster but more stable when agents can negotiate deadlocks and broadcast dropped-waste locations.

### 2) Green-to-Green Coordination

- **Run config:** `--green-coordination-values True,False`
- **Question:** does same-color arbitration reduce target conflicts?
- **Interpretation:** effect size on mean is small depending on the setting; variance changes are a bit more visible.

![Experience 2](batch_results/exp_green_coordination/plot_time_to_clear_vs_green_coordination.png)

This experiment complements global communication tests by isolating same-color coordination in zone `z1`. With `green_coordination=True`, visible green robots avoid pursuing the same waste target at the same time.
- In this setup, the mean improvement is not visible, but the variance is slightly smaller, which still suggests a consistency benefit from coordination. This may be because, with a limited vision range(3), this type of communication is used only rarely and, moreover, it affects only the processing of green waste. A more appropriate metric here might therefore be the time required to process green waste.

### 3) Red Agent Memory

- **Run config:** `--use-memory True,False`
- **Question:** does remembering disposal location reduce search overhead?
- **Interpretation:** memory-enabled settings are associated with lower mean completion time in the shown runs.

![Experience 3](batch_results/exp_memory/plot_time_to_clear_vs_use_memory.png)

As in the communication experiment, enabling memory improves outcomes, with an even larger effect size.

- Strong Reduction in Average Time: without memory (`False`), red agents must rediscover the disposal area repeatedly, and average completion time rises above 700 steps. With memory (`True`), the average drops near 200 steps.
- Large Consistency Gain: variance is also much lower with memory enabled, indicating more predictable end-to-end behavior.

### 4) Border Patrol

- **Run config:** `--patrol-border True,False`
- **Question:** does proactive border waiting improve handoffs?
- **Interpretation:** In this setting, `patrol_border` does not appear to have a strong effect on performance.

![Experience 4](batch_results/exp_patrol/plot_time_to_clear_vs_patrol_border.png)

- Enabling it slightly reduces the mean time to clear all waste, but the difference remains small compared with the variability across runs, and the error bars largely overlap. This suggests that border patrolling may provide at most a modest benefit here, rather than a clear improvement. A possible explanation is that, under this configuration, the agents already coordinate reasonably well through communication and memory, so the additional exploration structure brought by border patrolling has only a limited impact.

### 5) Initial Waste Distribution

- **Run config:** `--multiple-wastes True,False`
- **Question:** does starting with mixed waste types alter pipeline dynamics?
- **Interpretation:** mixed starts are often harder and can increase time and spread in this setup.

![Experience 5](batch_results/exp_multiple_wastes/plot_time_to_clear_vs_multiple_wastes.png)

As expected, enabling mixed initial waste types (`multiple_wastes=True`) increases both average completion time and variability.

- Increase in Average Time: homogeneous initialization (`False`) is faster on average (around 230 steps), while mixed initialization (`True`) is slower (around 330 steps).
- Increase in Variance: mixed starts also produce wider error bars, showing less consistent trajectories.

This is consistent with a harder cold-start pipeline, where all agent types must coordinate immediately in a more complex initial state.

### 6) Quantitative Scaling 

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


Interpretation of the scaling curves

- Green scaling: The number of robots is beneficial only up to an intermediate point. In the first graph, adding green robots strongly improves performance from 1 to about 4 robots, with a sharp decrease in the mean time-to-clear. Beyond that point, the gain disappears and performance slightly degrades, which suggests diminishing returns and possibly congestion or redundancy between agents. In other words, once enough green robots are available, the bottleneck likely shifts to another part of the task.

- Yellow scaling: the same pattern appears for yellow robots, with the best performance around 3 robots. Adding more yellow robots beyond this point increases the mean time-to-clear, which indicates that simply increasing team size does not necessarily improve coordination. An excess of robots of one type may create interference, while the limiting factor becomes the number of robots of the other roles or the structure of the environment.

- Red scaling: Same for the red ones. The plot suggests a non-monotonic effect. Increasing the number of red robots improves performance from 1 to about 3 robots, as shown by the decrease in mean time-to-clear. However, adding more red robots beyond this point does not provide further benefit and may even slightly worsen performance. 

- Vision scaling: Increasing the vision radius produces a clear improvement from 1 to about 3. Beyond that point, performance stabilizes and even slightly degrades for larger values, indicating that wider perception is useful only up to a certain threshold. A plausible explanation is that once robots can already detect relevant targets efficiently, further increasing vision does not significantly improve decision-making and may instead increase redundancy or coordination overhead.

- Waste scaling: as expected, higher initial waste increases mean time-to-clear.  For small to intermediate values, the time-to-clear remains relatively stable, which suggests that the system (with fixed params) can absorb a moderate increase in workload without a major loss in efficiency. However, beyond roughly 32–40 waste units, the curve rises more clearly, indicating that the system is reaching its capacity limits. In this regime, additional waste creates a heavier processing burden and the robots can no longer maintain the same level of efficiency.

### 7) Vision as Communication Fallback

- **Run config:** `--vision 1,2,3,4,5 --use-communication False`
- **Question:** how much can perception compensate when **messaging is disabled**?
- **Interpretation:** higher vision is associated with faster completion in this scenario.

![Experience 7](batch_results/exp_vision_no_comm/plot_time_to_clear_vs_vision.png)

As expected, increasing visual range significantly improves performance when communication is disabled.

- Decreased Average Time: with `comm=False`, agents rely entirely on perception; moving from vision `1` to higher values reduces mean completion time strongly.
- Reduced Variance: the 95% CI narrows as vision increases, indicating more stable outcomes.

This supports the idea that stronger local perception can partially compensate for missing wireless coordination.

### 8) Extreme Crowding Stress Test

- **Run config:** `--n-green-robots 10,15 --n-yellow-robots 8,12 --n-waste 16`
- **Question:** behavior under high contention and low resources.
- **Interpretation:** These results suggest a possible asymmetric effect of crowding across robot types.

![Experience 8 Green](batch_results/exp_extreme_crowding/plot_time_to_clear_vs_green_agents.png)
![Experience 8 Yellow](batch_results/exp_extreme_crowding/plot_time_to_clear_vs_yellow_agents.png)

Under this highly constrained setting, increasing the number of green robots appears to worsen performance, whereas increasing the number of yellow robots seems to slightly reduce the time-to-clear. A plausible explanation is that, with only one red robot, the system becomes bottlenecked at the final processing stage: adding more upstream green robots may increase congestion and overload the downstream pipeline, while additional yellow robots may help smooth intermediate transport. However, the confidence intervals are very large and only two values were tested for each variable, so this result should be interpreted as a tentative trend rather than strong evidence.

### 9) Lone-Wolf Baseline

- **Run config:** `--n-green-robots 1 --n-yellow-robots 1 --n-red-robots 1 --n-waste 16,32,64`
- **Question:** baseline throughput without same-color peer interactions.
- **Interpretation:** useful reference curve for comparing multi-robot configurations.

![Experience 9](batch_results/exp_lone_wolf/plot_time_to_clear_vs_waste.png)

As expected, increasing initial waste raises mean time-to-clear in the lone-wolf configuration.

- Near-Linear Trend: from `16 -> 64` waste units, completion time rises in a broadly linear fashion.
- Baseline Robustness: this run family provides a useful reference curve for comparing multi-robot speedups and crowding effects.
