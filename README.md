# Self-Organization of Robots in a Hostile Environment

**Group:** 14  
**Date:** 16 March 2026  
**Members:** Deodato V. Bastos Neto, Karina Musina  

---

## 1. Overview and Objectives
This project is an Agent-Based Model (ABM) developed to simulate a distributed, multi-agent system where autonomous robots must cooperate to clean a hostile, radioactive environment. 

The environment is decomposed into three zones (from west to east): Z1 (low radioactivity), Z2 (medium radioactivity), and Z3 (high radioactivity). Three types of robots (Green, Yellow, Red) are restricted to specific zones and must collect, transform, and transport waste eastward to a final disposal zone.

The objective of this simulation is to study **distributed problem solving**, where the global solution (a clean environment) emerges from local behaviors and interactions without a centralized controller.

---

## 2. Requirements & Running the Simulation

### Prerequisites
The project relies on Python 3+ and the Mesa framework with Solara for visualization.
```bash
pip install mesa solara matplotlib
```

### Execution
**1. Visual Mode (Web Interface):**
To launch the interactive SolaraViz dashboard, run:
```bash
solara run server.py
```

**2. Batch / Headless Mode:**
To run the simulation in the terminal and output the step-by-step metrics (useful for data collection):
```bash
python run.py --steps 100 --n-waste 30 --verbose
```

Example with Green-to-Green communication enabled (+ communication logs):
```bash
python run.py --steps 100 --n-waste 30 --green-coordination --log-communications --verbose
```


---

## 3. Theoretical Framework & M&S Scope

Following the Modeling & Simulation (M&S) theory principles:
* **Source System:** A hazardous waste management facility requiring specialized robotic handling.
* **Experimental Frame:** We observe the efficiency of the robots (measured by the count of red waste successfully put away) over a given number of steps, evaluating the impact of their movement mechanisms.
* **Model:** The `RobotMissionModel` acts as the arbiter, managing the spatial grid, spawning agents, and safely applying the consequences of actions. The process involves designing a model of a real system to understand its behavior and evaluate strategies.
* **Simulator:** Handled by the Mesa framework's execution engine.

---

## 4. System Properties 

In accordance with the properties of Multi-Agent Systems:

### Environment Properties
* **Partially Observable:** Robots do not possess a global map of all wastes. They can only perceive their current cell and immediate adjacent tiles. 
* **Dynamic:** The environment changes due to processes beyond a single agent's control (other agents are constantly picking up and dropping waste).
* **Discrete:** Space is represented as a discrete 2D grid, and time advances in discrete steps.
* **Stochastic (Initialization):** Wastes and robots are spawned at random coordinates. However, the agent deliberation process is mostly deterministic.

### Agent Properties
* **Autonomy:** Each robot operates independently. There is no global master planning their paths.
* **Loosely Coupled:** Agents do not have access to the internal states, memory, or code of other agents. They only interact indirectly via the environment.
* **Distributed Procedural Loop:** We enforce a strict procedural loop that interleaves the perceptions and actions of agents. Each class implements the procedural loop of the agent: percepts, deliberate, do.

---

## 5. Conceptual Choices & Architecture

### The Deliberation Constraint (`@staticmethod`)
To guarantee true encapsulation and autonomy, the `deliberate()` function in our agents is implemented as a `@staticmethod`. This forces the agent to make decisions *exclusively* based on its internal `knowledge` (memory/beliefs) and recent `percepts`. It physically prevents the agent from cheating by accessing `self.model` to gain a global view of the grid.

### Smart Pathfinding vs. Random Walk
The baseline requirement was to implement a random walk. However, a purely reactive random walk proved highly inefficient for the goals. We upgraded our agents to be **Cognitive Agents** by introducing a "look-ahead" mechanism:
1. The environment (`model.do`) passes `adjacent_tiles` in the percepts.
2. During deliberation, the agent scans these adjacent tiles.
3. If a target waste (or the disposal zone) is spotted, the agent decisively targets that cell instead of moving randomly.

### Green-to-Green Communication
We introduced an optional local communication protocol between Green robots, controlled by `green_coordination`.

When enabled, Green agents exchange two message types with visible Green peers:
* `green_visible_targets`: the waste targets currently visible to the sender.
* `green_state`: the sender position and current inventory state.

This supports two coordination behaviors:
1. **Shared-target arbitration:** if two Green agents see the same green waste, they agree on a single collector using a deterministic rule (closest robot, then agent id as tie-breaker).
2. **Inventory merge (1+1):** if two Green agents each carry exactly one green unit, they can coordinate so one transfers its unit to the other (`transfer_green`). This creates a `2 green` inventory on one robot, allowing faster `2 green -> 1 yellow` transformation.

Activation:
* **Solara UI:** `Enable Green Coordination` checkbox.
* **CLI (`run.py`):** `--green-coordination` / `--no-green-coordination`.
* **Batch (`batch_experiments.py`):** `--green-coordination` / `--no-green-coordination`.

Communication logs:
* Enable terminal logging with `log_communications` (`--log-communications` in CLI/batch or `Log Communications in Terminal` in Solara UI).

---
