from typing import Any, Dict, List, Optional, Tuple

from mesa import Model, Agent
from mesa.space import MultiGrid

from objects import Radioactivity, WasteDisposalZone, Waste
from agents import RobotAgent, GreenAgent, YellowAgent, RedAgent


Position = Tuple[int, int]


class RobotMissionModel(Model):
    def __init__(
        self,
        width: int,
        height: int,
        n_waste: int = 0,
        n_robots: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.grid: MultiGrid = MultiGrid(width, height, torus=False)

        # Environment agents
        self.radioactivity_agents: List[Radioactivity] = []
        self.waste_disposal_zone: Optional[WasteDisposalZone] = None
        self.waste_disposal_pos: Optional[Position] = None
        self.waste_agents: List[Waste] = []

        # Robot agents
        self.robot_agents: List[RobotAgent] = []

        # Build the static environment
        self._init_radioactivity_field()
        self._init_waste_disposal_zone()
        self._init_waste(n_waste)

        # Create robot agents
        self._init_robots(n_robots)

    def _zone_for_x(self, x: int) -> str:
        """Return 'z1', 'z2' or 'z3' according to the column index.

        The environment is decomposed into three vertical bands from west to
        east: z1 (low radioactivity), z2 (medium), z3 (high).
        """
        third = self.width / 3.0
        if x < third:
            return "z1"
        elif x < 2 * third:
            return "z2"
        else:
            return "z3"

    def _init_radioactivity_field(self) -> None:
        """Create one :class:`Radioactivity` agent per grid cell."""
        for x in range(self.width):
            zone = self._zone_for_x(x)
            for y in range(self.height):
                agent = Radioactivity(model=self, zone=zone)
                self.grid.place_agent(agent, (x, y))
                self.radioactivity_agents.append(agent)

    def _init_waste_disposal_zone(self) -> None:
        """Create a single waste-disposal cell on the eastern border."""
        x = self.width - 1
        y = int(self.rng.integers(0, self.height))
        zone = self._zone_for_x(x)
        agent = WasteDisposalZone(model=self, zone=zone)
        self.grid.place_agent(agent, (x, y))
        self.waste_disposal_zone = agent
        self.waste_disposal_pos = (x, y)

    def _init_waste(self, n_waste: int) -> None:
        """Create a number of waste objects at random positions."""
        # types = ("green", "yellow", "red")
        types = ("green",)
        for _ in range(n_waste):
            x = int(self.rng.integers(0, self.width // 3))
            y = int(self.rng.integers(0, self.height))
            waste_type = self.random.choice(types)
            agent = Waste(model=self, waste_type=waste_type)
            self.grid.place_agent(agent, (x, y))
            self.waste_agents.append(agent)

    def _init_robots(self, n_robots: int) -> None:
        """Create a number of robot agents at random positions."""
        types = ("green", "yellow", "red")
        for _ in range(n_robots):
            x = int(self.rng.integers(0, self.width // 3))
            y = int(self.rng.integers(0, self.height))
            robot_type = self.random.choice(types)

            if robot_type == "green":
                agent = GreenAgent(model=self)
            elif robot_type == "yellow":
                agent = YellowAgent(model=self)
            else:
                agent = RedAgent(model=self)

            self.grid.place_agent(agent, (x, y))
            self.robot_agents.append(agent)

    def step(self) -> None:
        self.agents.shuffle_do("step")

    def do(self, agent: RobotAgent, action: Any) -> Dict[Position, List[Any]]:
        """Execute an action in the environment and return percepts.

        Parameters
        ----------
        agent:
            The agent that intends to perform the action.
        action:
            A domain-specific description of the action. The first element
            (for a tuple/list) or the ``type``/``name`` attribute is used
            as the action type.

        Returns
        -------
        dict
            A mapping from positions to the list of objects present on each
            observable cell *after* executing the action (or, if the action
            was infeasible, without changing the state).
        """

        action_type = self._get_action_type(action)
        if action_type is None:
            raise ValueError(f"Cannot determine action type from {action!r}")

        if action_type == "move":
            target = self._get_move_target(agent, action)
            if self._is_move_feasible(agent, target):
                # self.grid.move_agent(agent, target)
                agent.apply_action({"type": "move", "to": target})

        elif action_type == "pick":
            if self._is_pick_feasible(agent):
                    # contents = self.grid.get_cell_list_contents([agent.pos])
                    # for obj in contents:
                    #     if getattr(obj, "waste_type", None) == agent.robot_type:
                    #         self.grid.remove_agent(obj)
                    #         self.waste_agents.remove(obj)
                    #         agent.knowledge['inventory'][agent.robot_type] += 1
                    #         break
                agent.apply_action({"type": "pickup"})

        elif action_type == "drop":
            if self._is_drop_feasible(agent):
                agent.apply_action({"type": "drop"})

        elif action_type == "transform":
            if self._is_transform_feasible(agent):
                agent.apply_action({"type": "transform"})
        else:
            pass

        return self._build_percepts(agent)

    @staticmethod
    def _get_action_type(action: Any) -> Optional[str]:
        """Extract an action type identifier from an action description.

        Supported conventions:
        - an object with a ``type`` attribute
        - an object with a ``name`` attribute
        - a tuple/list whose first element is the action type
        - a plain string
        """

        # Try attribute-based descriptions first
        if hasattr(action, "type"):
            return getattr(action, "type")  # type: ignore[no-any-return]
        if hasattr(action, "name"):
            return getattr(action, "name")  # type: ignore[no-any-return]

        # first element of a sequence, e.g. ("move", (x, y))
        if isinstance(action, (tuple, list)) and action:
            return str(action[0])

        if isinstance(action, str):
            return action

        return None

    def _get_move_target(self, agent: Agent, action: Any) -> Position:
        """Extract the target position from a MOVE action.

        Supported encodings:
        - ("move", (x, y))
        - {"type": "move", "to": (x, y)}
        """

        if isinstance(action, (tuple, list)) and len(action) >= 2:
            return tuple(action[1])  # type: ignore[return-value]
        if isinstance(action, dict) and "to" in action:
            return tuple(action["to"])  # type: ignore[return-value]

        possible_steps = self.grid.get_neighborhood(
            agent.pos,
            moore=True,
            include_center=False,
        )
        return self.random.choice(possible_steps)

    def _is_move_feasible(self, agent: Agent, target: Position) -> bool:
        """Check whether a MOVE action is feasible for the given agent.

        The MOVE action is feasible if:
        - the target position is a neighbouring cell, and
        - the target zone is accessible for the robot type:
          * green robots    -> only in z1
          * yellow robots   -> in z1 and z2
          * red robots      -> in z1, z2 and z3
        The robot type is expected to be available as ``agent.robot_type``
        with value in {"green", "yellow", "red"}.
        """

        # 1) Neighbourhood constraint
        neighbourhood = self.grid.get_neighborhood(
            agent.pos, moore=True, include_center=False
        )
        if target not in neighbourhood:
            return False

        # 2) Zone-access constraint according to robot type
        zone = self._zone_for_x(target[0])
        robot_type = getattr(agent, "robot_type", None)

        if robot_type == "green":
            return zone == "z1"
        if robot_type == "yellow":
            return zone in {"z1", "z2"}
        if robot_type == "red":
            return zone in {"z1", "z2", "z3"}

        return False

    def _is_pick_feasible(self, agent: RobotAgent) -> bool:
        """Check whether a PICK action is feasible for the given agent.

        The PICK action is feasible if there is at least one waste object of the same color on
        the same cell as the agent.
        """

        contents = self.grid.get_cell_list_contents([agent.pos])
        for obj in contents:
            if getattr(obj, "waste_type", None) is not None:
                return obj.waste_type == agent.type
        return False

    def _is_transform_feasible(self, agent: RobotAgent) -> bool:
        """Check whether a TRANSFORM action is feasible for the given agent.

        The TRANSFORM action is feasible if the agent has two item of the same color in its inventory.
        """

        inventory = agent.knowledge.get("inventory", {})
        return inventory.get(agent.type, 0) >= 2

    def _is_drop_feasible(self, agent: RobotAgent) -> bool:
        """Check whether a DROP action is feasible for the given agent.

        The DROP action is feasible if:
        - the agent is on the waste-disposal cell
        - and the agent has at least one red waste item in its inventory.
        - or the agent is on the frontier cell to the next zone, and it has one waste item of the same color as the next zone in its inventory
        - and the agent cannt move east
        """

        if self.waste_disposal_pos is None:
            return False

        if agent.knowledge["inventory"]["red"] > 0 and agent.pos != self.waste_disposal_pos:
            return False

        inventory = agent.knowledge.get("inventory", {})
        if agent.type == "green":
            next_zone = "z2"
            next_zone_type = "yellow"
        elif agent.type == "yellow":
            next_zone = "z3"
            next_zone_type = "red"

        return self._zone_for_x(agent.pos[0]) == next_zone and inventory.get(next_zone_type, 0) > 0


    def _build_percepts(self, agent: Agent) -> Dict[Position, List[Any]]:
        """Return what the agent can perceive after an action.

        For simplicity, we return the contents of the current cell and its
        neighbouring cells as a dictionary:

        ``{ position: [objects in that cell], ... }``
        """

        percepts: Dict[Position, List[Any]] = {}
        cells = self.grid.get_neighborhood(
            agent.pos, moore=True, include_center=True
        )
        for pos in cells:
            contents = self.grid.get_cell_list_contents([pos])
            percepts[pos] = list(contents)
        return percepts
