"""
Group: 14
Date: 20 March 2026
Members: Deodato V. Bastos Neto, Karina Musina
"""

import matplotlib.patches as patches
from mesa.visualization import SolaraViz, make_plot_component, make_space_component
from mesa.visualization.components import AgentPortrayalStyle
from mesa.datacollection import DataCollector
from typing import Dict
import math
import pandas as pd
import random
import threading

from agents import GreenAgent, RedAgent, YellowAgent
from model import RobotMissionModel
from objects import Radioactivity, Waste, WasteDisposalZone

# ── Frame-level collectors ────────────────────────────────────────────────────
# agent_portrayal() is called for ALL agents before post_process() runs,
# so we can safely populate these lists here and consume them there.
_robots = []
_disposal = []
_current_model = None


_ORIGINAL_GET_MODEL_VARS_DATAFRAME = DataCollector.get_model_vars_dataframe


def _patched_get_model_vars_dataframe(self):
    model = getattr(self, "model", None)
    lock = getattr(model, "_datacollector_lock", None)

    if lock is None:
        try:
            return _ORIGINAL_GET_MODEL_VARS_DATAFRAME(self)
        except ValueError:
            snapshot = {name: list(values) for name, values in self.model_vars.items()}
    else:
        with lock:
            snapshot = {name: list(values) for name, values in self.model_vars.items()}

    if not snapshot:
        return pd.DataFrame()

    min_len = min(len(values) for values in snapshot.values())
    max_len = max(len(values) for values in snapshot.values())
    if min_len != max_len:
        snapshot = {name: values[:min_len] for name, values in snapshot.items()}

    return pd.DataFrame(snapshot)


if not getattr(DataCollector, "_threadsafe_model_vars_patch_applied", False):
    DataCollector.get_model_vars_dataframe = _patched_get_model_vars_dataframe
    DataCollector._threadsafe_model_vars_patch_applied = True


def _robot_color(agent):
    if isinstance(agent, GreenAgent):
        return "#004B23"
    if isinstance(agent, YellowAgent):
        return "#FF9F1C"
    if isinstance(agent, RedAgent):
        return "#9E0059"
    return "tab:blue"


def _robot_carry_count(agent):
    knowledge = getattr(agent, "knowledge", {})
    if not isinstance(knowledge, dict):
        return 0
    inventory = knowledge.get("inventory", {})
    if not isinstance(inventory, dict):
        return 0
    return sum(int(inventory.get(k, 0) or 0) for k in ("green", "yellow", "red"))


def _remaining_counts(model) -> Dict[str, int]:
    counts = {"green": 0, "yellow": 0, "red": 0}

    for waste in getattr(model, "waste_agents", []):
        waste_type = getattr(waste, "waste_type", None)
        if waste_type in counts:
            counts[waste_type] += 1

    for robot in getattr(model, "robot_agents", []):
        knowledge = getattr(robot, "knowledge", {})
        if not isinstance(knowledge, dict):
            continue
        inventory = knowledge.get("inventory", {})
        if not isinstance(inventory, dict):
            continue
        counts["green"] += int(inventory.get("green", 0) or 0)
        counts["yellow"] += int(inventory.get("yellow", 0) or 0)
        counts["red"] += int(inventory.get("red", 0) or 0)

    return counts


def _remaining_weight_for(model, color: str) -> int:
    weights = {"green": 1, "yellow": 2, "red": 4}
    counts = _remaining_counts(model)
    return int(counts.get(color, 0)) * int(weights[color])


def _remaining_weight_total(model) -> int:
    counts = _remaining_counts(model)
    return (
        int(counts["green"]) * 1
        + int(counts["yellow"]) * 2
        + int(counts["red"]) * 4
    )


def _is_mission_complete(model) -> bool:
    return _remaining_weight_total(model) == 0


def _draw_fireworks(ax, model) -> None:
    width = max(1, int(getattr(model, "width", 20)))
    height = max(1, int(getattr(model, "height", 10)))
    phase = float(getattr(model, "_fireworks_phase", 0))
    rng = random.Random(2026)
    colors = ["#FF595E", "#FFCA3A", "#8AC926", "#1982C4", "#6A4C93", "#F15BB5"]

    for i in range(8):
        # Each burst has a short life so particles appear/disappear naturally.
        burst_period = 32.0
        burst_life = 20.0
        t = (phase - i * 4.0) % burst_period
        if t > burst_life:
            continue

        progress = t / burst_life  # 0 -> 1 during burst
        fade = math.sin(math.pi * progress)  # appear then disappear

        base_x = rng.uniform(1, max(1.5, width - 2))
        base_y = rng.uniform(1, max(1.5, height - 2))
        cx = base_x + 0.15 * math.cos(phase * 0.08 + i)
        cy = base_y + 0.12 * math.sin(phase * 0.09 + i * 0.7)
        color = rng.choice(colors)
        core_size = 70 + 260 * fade
        ax.scatter(
            [cx],
            [cy],
            c=[color],
            marker="*",
            s=core_size,
            zorder=12,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.25 + 0.75 * fade,
        )

        radius = (0.25 + 1.4 * progress) * rng.uniform(0.8, 1.15)
        points = 12
        rot = phase * 0.12 + i * 0.5
        xs = [cx + math.cos((2 * math.pi * j) / points + rot) * radius for j in range(points)]
        ys = [cy + math.sin((2 * math.pi * j) / points + rot) * radius for j in range(points)]
        ax.scatter(
            xs,
            ys,
            c=[color] * points,
            marker="o",
            s=14 + 20 * fade,
            zorder=11,
            alpha=0.15 + 0.75 * fade,
            edgecolors="none",
        )


def _ensure_remaining_weight_reporters(model) -> None:
    dc = getattr(model, "datacollector", None)
    if dc is None:
        return

    reporters = {
        "remaining_green_weight": lambda m: _remaining_weight_for(m, "green"),
        "remaining_yellow_weight": lambda m: _remaining_weight_for(m, "yellow"),
        "remaining_red_weight": lambda m: _remaining_weight_for(m, "red"),
        "remaining_weighted_total": _remaining_weight_total,
    }

    for name, reporter in reporters.items():
        if name not in dc.model_reporters:
            dc._new_model_reporter(name, reporter)

        existing_len = len(dc.model_vars.get(name, []))
        target_len = 0
        for values in dc.model_vars.values():
            target_len = max(target_len, len(values))

        if existing_len < target_len:
            value_now = reporter(model)
            dc.model_vars[name].extend([value_now] * (target_len - existing_len))
        elif existing_len > target_len:
            dc.model_vars[name] = dc.model_vars[name][:target_len]


_ORIGINAL_ROBOTMISSION_INIT = RobotMissionModel.__init__
_ORIGINAL_ROBOTMISSION_STEP = RobotMissionModel.step


def _patched_robotmission_init(
    self,
    width: int,
    height: int,
    n_waste: int = 30,
    n_robots: int = 0,
    n_green_robots: int = 3,
    n_yellow_robots: int = 2,
    n_red_robots: int = 1,
    vision: int = 1,
    green_coordination: bool = True,
    log_communications: bool = False,
    use_memory: bool = True,
    patrol_border: bool = False,
    use_communication: bool = True,
    multiple_wastes: bool = False,
    rng=None,
    seed=None,
) -> None:
    _ORIGINAL_ROBOTMISSION_INIT(
        self,
        width=width,
        height=height,
        n_waste=n_waste,
        n_robots=n_robots,
        n_green_robots=n_green_robots,
        n_yellow_robots=n_yellow_robots,
        n_red_robots=n_red_robots,
        vision=vision,
        green_coordination=green_coordination,
        log_communications=log_communications,
        use_memory=use_memory,
        patrol_border=patrol_border,
        use_communication=use_communication,
        multiple_wastes=multiple_wastes,
        rng=rng,
        seed=seed,
    )
    self._datacollector_lock = threading.RLock()
    _ensure_remaining_weight_reporters(self)
    self._viz_completion_animation = True
    self._fireworks_phase = 0


def _patched_robotmission_step(self) -> None:
    lock = getattr(self, "_datacollector_lock", None)
    if lock is None:
        # Fallback if lock is missing for any reason.
        lock = threading.RLock()
        self._datacollector_lock = lock

    with lock:
        # Freeze world once complete, but keep stepping so fireworks can animate.
        if (
            getattr(self, "_viz_completion_animation", False)
            and getattr(self, "time_to_clear", None) is not None
        ):
            self._fireworks_phase = int(getattr(self, "_fireworks_phase", 0)) + 1
            self.running = True
            # Keep time and history frozen after completion.
            return

        _ORIGINAL_ROBOTMISSION_STEP(self)

        if (
            getattr(self, "_viz_completion_animation", False)
            and getattr(self, "time_to_clear", None) is not None
        ):
            self.running = True
            self._fireworks_phase = int(getattr(self, "_fireworks_phase", 0)) + 1


if not getattr(RobotMissionModel, "_remaining_viz_patch_applied", False):
    RobotMissionModel.__init__ = _patched_robotmission_init
    RobotMissionModel.step = _patched_robotmission_step
    RobotMissionModel._remaining_viz_patch_applied = True


def agent_portrayal(agent):
    global _current_model
    _current_model = getattr(agent, "model", None)

    # 1. Radioactivity background — invisible, isolated marker
    if isinstance(agent, Radioactivity):
        return AgentPortrayalStyle(
            color="white", marker="o", size=0,
            alpha=0.0, zorder=0,
            edgecolors="white", linewidths=0.0,
        )

    # 2. Disposal zone — collect for manual drawing
    if isinstance(agent, WasteDisposalZone):
        _disposal.append(agent)
        return AgentPortrayalStyle(
            color="white", marker="o", size=0,
            alpha=0.0, zorder=0,
            edgecolors="white", linewidths=0.0,
        )

    # 3. Waste — rendered by Mesa's scatter (circles work fine)
    if isinstance(agent, Waste):
        waste_type = getattr(agent, "waste_type", "")
        color = {"green": "#2A9D8F", "yellow": "#E9C46A",
                 "red": "#D62828"}.get(waste_type, "gray")
        return AgentPortrayalStyle(
            color=color, marker="o", size=80,
            alpha=1.0, zorder=4,
            edgecolors="black", linewidths=1.5,
        )

    # 4. Robots — collect for manual drawing
    _robots.append(agent)
    return AgentPortrayalStyle(
        color="white", marker="o", size=0,
        alpha=0.0, zorder=0,
        edgecolors="white", linewidths=0.0,
    )


def draw_background_zones(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    grid_width = int(round(xlim[1] + 0.5))
    grid_height = int(round(ylim[1] + 0.5))
    third = grid_width / 3.0

    for x in range(grid_width):
        color = "#8FD694" if x < third else (
            "#FFD166" if x < 2 * third else "#FF6B6B")
        ax.add_patch(patches.Rectangle(
            (x - 0.5, -0.5), 1.0, grid_height,
            facecolor=color, alpha=0.4, zorder=0, edgecolor="none",
        ))


def post_process(ax):
    draw_background_zones(ax)

    # Draw disposal zones as stars
    for agent in _disposal:
        x, y = agent.pos
        ax.scatter([x], [y], c=["black"], marker="*",
                   s=200, zorder=5, edgecolors="white", linewidths=1.5)

    # Draw robots as diamonds
    for agent in _robots:
        x, y = agent.pos
        ax.scatter([x], [y], c=[_robot_color(agent)], marker="D",
                   s=140, zorder=6, edgecolors="white", linewidths=2.0)
        ax.text(
            x,
            y,
            str(_robot_carry_count(agent)),
            color="white",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=7,
        )

    if _current_model is not None:
        current_time = getattr(_current_model, "time", 0.0)
        cumulative_moves = getattr(_current_model, "cumulative_moves", 0)
        ax.text(
            0.01,
            0.99,
            f"time={current_time:.0f} | cumulative_moves={int(cumulative_moves)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "black"},
            zorder=10,
        )
        if _is_mission_complete(_current_model):
            _draw_fireworks(ax, _current_model)
            pulse = 0.5 + 0.5 * math.sin(
                float(getattr(_current_model, "_fireworks_phase", 0)) * 0.25
            )
            ax.text(
                0.5,
                0.5,
                "MISSION COMPLETE",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14 + 6 * pulse,
                fontweight="bold",
                color="#1B4332",
                bbox={
                    "facecolor": "#D8F3DC",
                    "alpha": 0.7 + 0.25 * pulse,
                    "edgecolor": "#1B4332",
                },
                zorder=13,
            )

    # Clear collectors for next frame
    _robots.clear()
    _disposal.clear()


def post_process_waste_plot(ax):
    ax.set_title("Remaining Weighted Waste by Type (Grid + Inventory)")
    ax.set_ylabel("Weighted waste")
    ax.grid(alpha=0.3)


model_params = {
    "width":          {"type": "SliderInt", "value": 20,  "label": "Grid width",        "min": 10,  "max": 60,  "step": 1},
    "height":         {"type": "SliderInt", "value": 10,  "label": "Grid height",       "min": 5,   "max": 40,  "step": 1},
    "n_waste":        {"type": "SliderInt", "value": 32,  "label": "Initial green waste", "min": 4,   "max": 200, "step": 4},
    "n_green_robots": {"type": "SliderInt", "value": 3,   "label": "Green robots",      "min": 0,   "max": 30,  "step": 1},
    "n_yellow_robots": {"type": "SliderInt", "value": 2,   "label": "Yellow robots",     "min": 0,   "max": 30,  "step": 1},
    "n_red_robots":   {"type": "SliderInt", "value": 1,   "label": "Red robots",        "min": 0,   "max": 30,  "step": 1},
    "vision":         {"type": "SliderInt", "value": 2,   "label": "Robot Vision Radius", "min": 1, "max": 5, "step": 1},
    "green_coordination": {"type": "Checkbox", "value": True, "label": "Enable Green Coordination"},
    "log_communications": {"type": "Checkbox", "value": False, "label": "Log Communications in Terminal"},
    "use_memory":     {"type": "Checkbox",  "value": True, "label": "Use Red Robot Memory"},
    "patrol_border":  {"type": "Checkbox",  "value": True, "label": "Enable Border Patrol"},
    "use_communication": {"type": "Checkbox",  "value": True, "label": "Enable Robot Communication"},
    "multiple_wastes": {"type": "Checkbox",  "value": False, "label": "Use Multiple Waste Types"},
    "seed": 42,
}

space_component = make_space_component(
    agent_portrayal, post_process=post_process)
waste_plot_component = make_plot_component(
    {
        "remaining_green_weight": "#2A9D8F",
        "remaining_yellow_weight": "#E9C46A",
        "remaining_red_weight": "#D62828",
        "remaining_weighted_total": "#111111",
    },
    post_process=post_process_waste_plot,
)

page = SolaraViz(
    RobotMissionModel(
        width=20,
        height=10,
        n_waste=32,
        n_green_robots=3,
        n_yellow_robots=2,
        n_red_robots=1,
        vision=2,
        green_coordination=True,
        log_communications=False,
        use_memory=True,
        patrol_border=True,
        use_communication=True,
        multiple_wastes=False,
        seed=42,
    ),
    components=[space_component, waste_plot_component],
    model_params=model_params,
    name="Robot Mission",
)
