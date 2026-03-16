import mesa
import solara
from matplotlib.figure import Figure
from mesa.visualization.utils import update_counter
from mesa.visualization.components import AgentPortrayalStyle
from mesa.visualization import SolaraViz, make_plot_component, make_space_component


from model import RobotMissionModel

def agent_portrayal(agent):
    size = 10
    color = "tab:red"
    if agent.wealth > 0:
        size = 50
        color = "tab:blue"
    return AgentPortrayalStyle(size=size, color=color)

model_params = {
    "n": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of waste items:",
        "min": 10,
        "max": 100,
        "step": 1,
    },
    "width": 10,
    "height": 10,
}

model = RobotMissionModel(50, 10, 10)

SpaceGraph = make_space_component(agent_portrayal)

page = SolaraViz(
    model,
    components=[SpaceGraph],
    model_params=model_params,
    name="Boltzmann Wealth Model",
)

page