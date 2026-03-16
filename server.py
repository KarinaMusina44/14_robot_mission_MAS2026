from mesa.visualization import SolaraViz, make_space_component
from mesa.visualization.components import AgentPortrayalStyle

from agents import GreenAgent, RedAgent, YellowAgent
from model import RobotMissionModel
from objects import Radioactivity, Waste, WasteDisposalZone


def agent_portrayal(agent):
    if isinstance(agent, Radioactivity):
        zone = getattr(agent, "zone", None)
        if zone == "z1":
            color = "#8FD694"
        elif zone == "z2":
            color = "#FFD166"
        else:
            color = "#FF6B6B"
        return AgentPortrayalStyle(
            color=color,
            marker="s",
            size=220,
            alpha=0.16,
            zorder=0,
            edgecolors="none",
            linewidths=0.0,
        )

    if isinstance(agent, WasteDisposalZone):
        return AgentPortrayalStyle(
            color="black",
            marker="P",
            size=180,
            alpha=1.0,
            zorder=1,
            edgecolors="white",
            linewidths=1.2,
        )

    if isinstance(agent, Waste):
        waste_type = getattr(agent, "waste_type", "")
        if waste_type == "green":
            color = "#2A9D8F"
        elif waste_type == "yellow":
            color = "#E9C46A"
        elif waste_type == "red":
            color = "#D62828"
        else:
            color = "gray"
        return AgentPortrayalStyle(
            color=color,
            marker="o",
            size=70,
            alpha=1.0,
            zorder=1,
            edgecolors="black",
            linewidths=0.6,
        )

    if isinstance(agent, GreenAgent):
        color = "#26412F"
    elif isinstance(agent, YellowAgent):
        color = "#F4A300"
    elif isinstance(agent, RedAgent):
        color = "#B00020"
    else:
        color = "tab:blue"

    return AgentPortrayalStyle(
        color=color,
        marker="^",
        size=105,
        alpha=1.0,
        zorder=1,
        edgecolors="white",
        linewidths=1.0,
    )


model_params = {
    "width": {"type": "SliderInt", "value": 20, "label": "Grid width", "min": 10, "max": 60, "step": 1},
    "height": {"type": "SliderInt", "value": 10, "label": "Grid height", "min": 5, "max": 40, "step": 1},
    "n_waste": {"type": "SliderInt", "value": 30, "label": "Initial green waste", "min": 0, "max": 200, "step": 1},
    "n_green_robots": {"type": "SliderInt", "value": 3, "label": "Green robots", "min": 0, "max": 30, "step": 1},
    "n_yellow_robots": {"type": "SliderInt", "value": 2, "label": "Yellow robots", "min": 0, "max": 30, "step": 1},
    "n_red_robots": {"type": "SliderInt", "value": 1, "label": "Red robots", "min": 0, "max": 30, "step": 1},
    "seed": 42,
}

model = RobotMissionModel(
    width=20,
    height=10,
    n_waste=30,
    n_green_robots=3,
    n_yellow_robots=2,
    n_red_robots=1,
    seed=42,
)

space_component = make_space_component(agent_portrayal)

page = SolaraViz(
    model,
    components=[space_component],
    model_params=model_params,
    name="Robot Mission",
)
