import matplotlib.patches as patches
from mesa.visualization import SolaraViz, make_plot_component, make_space_component
from mesa.visualization.components import AgentPortrayalStyle

from agents import GreenAgent, RedAgent, YellowAgent
from model import RobotMissionModel
from objects import Radioactivity, Waste, WasteDisposalZone

# ── Frame-level collectors ────────────────────────────────────────────────────
# agent_portrayal() is called for ALL agents before post_process() runs,
# so we can safely populate these lists here and consume them there.
_robots = []
_disposal = []
_current_model = None


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

    # Clear collectors for next frame
    _robots.clear()
    _disposal.clear()


def post_process_moves_plot(ax):
    ax.set_title("Cumulative Moves by Robot Color")
    ax.set_ylabel("Cumulative moves")
    ax.grid(alpha=0.3)


model_params = {
    "width":          {"type": "SliderInt", "value": 20,  "label": "Grid width",        "min": 10,  "max": 60,  "step": 1},
    "height":         {"type": "SliderInt", "value": 10,  "label": "Grid height",       "min": 5,   "max": 40,  "step": 1},
    "n_waste":        {"type": "SliderInt", "value": 30,  "label": "Initial green waste", "min": 0,   "max": 200, "step": 1},
    "n_green_robots": {"type": "SliderInt", "value": 3,   "label": "Green robots",      "min": 0,   "max": 30,  "step": 1},
    "n_yellow_robots": {"type": "SliderInt", "value": 2,   "label": "Yellow robots",     "min": 0,   "max": 30,  "step": 1},
    "n_red_robots":   {"type": "SliderInt", "value": 1,   "label": "Red robots",        "min": 0,   "max": 30,  "step": 1},
    "rng": 42,
}

model = RobotMissionModel(width=20, height=10, n_waste=30,
                          n_green_robots=3, n_yellow_robots=2,
                          n_red_robots=1, rng=42)

space_component = make_space_component(
    agent_portrayal, post_process=post_process)
moves_plot_component = make_plot_component(
    {
        "cumulative_moves": "#111111",
        "cumulative_moves_green": "#004B23",
        "cumulative_moves_yellow": "#FF9F1C",
        "cumulative_moves_red": "#9E0059",
    },
    post_process=post_process_moves_plot,
)

page = SolaraViz(
    model,
    components=[space_component, moves_plot_component],
    model_params=model_params,
    name="Robot Mission",
)
