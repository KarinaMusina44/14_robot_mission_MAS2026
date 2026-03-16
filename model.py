from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Optional, Tuple

from mesa import Agent, Model
from mesa.space import MultiGrid

from agents import GreenAgent, RobotAgent, RedAgent, YellowAgent
from objects import Radioactivity, Waste, WasteDisposalZone


Position = Tuple[int, int]


class RobotMissionModel(Model):
    def __init__(
        self,
        width: int,
        height: int,
        n_waste: int = 30,
        n_robots: int = 0,
        n_green_robots: int = 3,
        n_yellow_robots: int = 2,
        n_red_robots: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.grid: MultiGrid = MultiGrid(width, height, torus=False)

        self.radioactivity_agents: List[Radioactivity] = []
        self.waste_agents: List[Waste] = []
        self.robot_agents: List[RobotAgent] = []
        self.waste_disposal_zone: Optional[WasteDisposalZone] = None
        self.waste_disposal_pos: Optional[Position] = None
        self.disposed_counts: Dict[str, int] = {"green": 0, "yellow": 0, "red": 0}

        self._init_radioactivity_field()
        self._init_waste_disposal_zone()
        self._init_waste(n_waste)
        self._init_robots(
            n_robots=n_robots,
            n_green_robots=n_green_robots,
            n_yellow_robots=n_yellow_robots,
            n_red_robots=n_red_robots,
        )

    def _zone_for_x(self, x: int) -> str:
        third = self.width / 3.0
        if x < third:
            return "z1"
        if x < 2 * third:
            return "z2"
        return "z3"

    def _init_radioactivity_field(self) -> None:
        for x in range(self.width):
            zone = self._zone_for_x(x)
            for y in range(self.height):
                obj = Radioactivity(model=self, zone=zone)
                self.grid.place_agent(obj, (x, y))
                self.radioactivity_agents.append(obj)

    def _init_waste_disposal_zone(self) -> None:
        x = self.width - 1
        y = int(self.rng.integers(0, self.height))
        obj = WasteDisposalZone(model=self, zone="disposal")
        self.grid.place_agent(obj, (x, y))
        self.waste_disposal_zone = obj
        self.waste_disposal_pos = (x, y)

    def _init_waste(self, n_waste: int) -> None:
        # Step 1 starts with initial green waste in z1.
        for _ in range(max(0, n_waste)):
            x = int(self.rng.integers(0, max(1, self.width // 3)))
            y = int(self.rng.integers(0, self.height))
            obj = Waste(model=self, waste_type="green")
            self.grid.place_agent(obj, (x, y))
            self.waste_agents.append(obj)

    def _init_robots(
        self,
        n_robots: int,
        n_green_robots: int,
        n_yellow_robots: int,
        n_red_robots: int,
    ) -> None:
        if n_robots > 0 and (n_green_robots + n_yellow_robots + n_red_robots) == 0:
            for _ in range(n_robots):
                robot_cls = self.random.choice([GreenAgent, YellowAgent, RedAgent])
                self._spawn_one_robot(robot_cls)
            return

        for _ in range(max(0, n_green_robots)):
            self._spawn_one_robot(GreenAgent)
        for _ in range(max(0, n_yellow_robots)):
            self._spawn_one_robot(YellowAgent)
        for _ in range(max(0, n_red_robots)):
            self._spawn_one_robot(RedAgent)

    def _spawn_one_robot(self, robot_cls: type[RobotAgent]) -> None:
        robot = robot_cls(model=self)
        pos = self._random_position_in_zones(robot.allowed_zones)
        self.grid.place_agent(robot, pos)
        self.robot_agents.append(robot)

    def _random_position_in_zones(self, zones: set[str]) -> Position:
        candidates: List[Position] = []
        for x in range(self.width):
            if self._zone_for_x(x) not in zones:
                continue
            for y in range(self.height):
                candidates.append((x, y))
        return self.random.choice(candidates) if candidates else (0, 0)

    def step(self) -> None:
        robots = list(self.robot_agents)
        self.random.shuffle(robots)
        for robot in robots:
            robot.step()

    def do(self, agent: RobotAgent, action: Any) -> Dict[str, Any]:
        action_type = self._get_action_type(action)
        inventory = self._get_inventory(agent)
        action_success = False

        if action_type == "move":
            target = self._get_move_target(action)
            if target is not None and self._is_move_feasible(agent, target):
                self.grid.move_agent(agent, target)
                action_success = True

        elif action_type == "move_random":
            moves = self._allowed_moves_for(agent)
            if moves:
                self.grid.move_agent(agent, self.random.choice(moves))
                action_success = True

        elif action_type == "move_east":
            east_moves = self._east_moves_for(agent)
            if east_moves:
                self.grid.move_agent(agent, self.random.choice(east_moves))
                action_success = True
            else:
                fallback_moves = self._allowed_moves_for(agent)
                if fallback_moves:
                    self.grid.move_agent(agent, self.random.choice(fallback_moves))
                    action_success = True

        elif action_type == "pickup":
            waste_type = self._action_get(action, "waste")
            if waste_type in {"green", "yellow", "red"} and self.remove_one_waste_at(
                agent.pos, waste_type
            ):
                inventory[waste_type] += 1
                action_success = True

        elif action_type == "transform":
            src = self._action_get(action, "from")
            dst = self._action_get(action, "to")
            count = int(self._action_get(action, "count", 0) or 0)
            if src in inventory and dst in inventory and count > 0 and inventory[src] >= count:
                inventory[src] -= count
                inventory[dst] += 1
                action_success = True

        elif action_type == "drop":
            waste_type = self._action_get(action, "waste")
            if (
                waste_type in inventory
                and inventory[waste_type] > 0
                and self._is_drop_feasible(agent)
            ):
                inventory[waste_type] -= 1
                self.add_one_waste_at(agent.pos, waste_type)
                action_success = True

        elif action_type == "put_away":
            waste_type = self._action_get(action, "waste")
            if (
                waste_type in inventory
                and inventory[waste_type] > 0
                and self._is_disposal_cell(agent.pos)
            ):
                inventory[waste_type] -= 1
                self.disposed_counts[waste_type] += 1
                action_success = True

        elif action_type == "wait":
            action_success = True

        return self._build_percepts(agent, action_success)

    @staticmethod
    def _action_get(action: Any, key: str, default: Any = None) -> Any:
        if isinstance(action, dict):
            return action.get(key, default)
        return getattr(action, key, default)

    @staticmethod
    def _get_action_type(action: Any) -> Optional[str]:
        if isinstance(action, dict):
            action_type = action.get("type")
            return str(action_type) if action_type is not None else None
        if hasattr(action, "type"):
            return str(getattr(action, "type"))
        if hasattr(action, "name"):
            return str(getattr(action, "name"))
        if isinstance(action, (tuple, list)) and action:
            return str(action[0])
        if isinstance(action, str):
            return action
        return None

    def _get_move_target(self, action: Any) -> Optional[Position]:
        if isinstance(action, dict) and "to" in action:
            return tuple(action["to"])  # type: ignore[return-value]
        if isinstance(action, (tuple, list)) and len(action) >= 2:
            return tuple(action[1])  # type: ignore[return-value]
        return None

    def _allowed_moves_for(self, agent: RobotAgent) -> List[Position]:
        neighbors = self.grid.get_neighborhood(agent.pos, moore=False, include_center=False)
        return [p for p in neighbors if self._zone_for_x(p[0]) in agent.allowed_zones]

    def _east_moves_for(self, agent: RobotAgent) -> List[Position]:
        return [p for p in self._allowed_moves_for(agent) if p[0] > agent.pos[0]]

    def _is_move_feasible(self, agent: RobotAgent, target: Position) -> bool:
        return target in self._allowed_moves_for(agent)

    def _is_drop_feasible(self, agent: RobotAgent) -> bool:
        target_zone = getattr(agent, "next_zone_for_drop", None)
        if not isinstance(target_zone, str):
            return False
        for p in self.grid.get_neighborhood(agent.pos, moore=False, include_center=False):
            if p[0] > agent.pos[0] and self._zone_for_x(p[0]) == target_zone:
                return True
        return False

    def _is_disposal_cell(self, pos: Position) -> bool:
        if self.waste_disposal_pos is not None and pos == self.waste_disposal_pos:
            return True
        return any(isinstance(obj, WasteDisposalZone) for obj in self.grid.get_cell_list_contents([pos]))

    def _get_inventory(self, agent: RobotAgent) -> Dict[str, int]:
        knowledge = getattr(agent, "knowledge", {})
        inventory = knowledge.get("inventory")
        if not isinstance(inventory, dict):
            knowledge["inventory"] = {"green": 0, "yellow": 0, "red": 0}
            inventory = knowledge["inventory"]
        for k in ("green", "yellow", "red"):
            inventory.setdefault(k, 0)
        return inventory

    def add_one_waste_at(self, pos: Position, waste_type: str) -> bool:
        if waste_type not in {"green", "yellow", "red"}:
            return False
        obj = Waste(model=self, waste_type=waste_type)
        self.grid.place_agent(obj, pos)
        self.waste_agents.append(obj)
        return True

    def remove_one_waste_at(self, pos: Position, waste_type: str) -> bool:
        for obj in self.grid.get_cell_list_contents([pos]):
            if isinstance(obj, Waste) and obj.waste_type == waste_type:
                self.grid.remove_agent(obj)
                with contextlib.suppress(ValueError):
                    self.waste_agents.remove(obj)
                with contextlib.suppress(Exception):
                    obj.remove()
                return True
        return False

    def _cell_wastes(self, pos: Position) -> Dict[str, int]:
        counts = {"green": 0, "yellow": 0, "red": 0}
        for obj in self.grid.get_cell_list_contents([pos]):
            waste_type = getattr(obj, "waste_type", None)
            if waste_type in counts:
                counts[waste_type] += 1
        return counts

    def _adjacent_tiles_percepts(self, agent: RobotAgent) -> Dict[Position, Dict[str, Any]]:
        data: Dict[Position, Dict[str, Any]] = {}
        cells = self.grid.get_neighborhood(agent.pos, moore=True, include_center=True)
        for pos in cells:
            contents = self.grid.get_cell_list_contents([pos])
            data[pos] = {
                "zone": self._zone_for_x(pos[0]),
                "is_disposal_zone": self._is_disposal_cell(pos),
                "wastes": self._cell_wastes(pos),
                "contents": [obj.__class__.__name__ for obj in contents],
            }
        return data

    def _build_percepts(self, agent: RobotAgent, action_success: bool) -> Dict[str, Any]:
        next_zone = getattr(agent, "next_zone_for_drop", None)
        frontier_to_next_zone = False
        if isinstance(next_zone, str):
            for pos in self.grid.get_neighborhood(agent.pos, moore=False, include_center=False):
                if pos[0] > agent.pos[0] and self._zone_for_x(pos[0]) == next_zone:
                    frontier_to_next_zone = True
                    break

        return {
            "position": tuple(agent.pos),
            "zone": self._zone_for_x(agent.pos[0]),
            "cell_wastes": self._cell_wastes(agent.pos),
            "allowed_moves": self._allowed_moves_for(agent),
            "in_disposal_zone": self._is_disposal_cell(agent.pos),
            "frontier_to_next_zone": frontier_to_next_zone,
            "inventory": dict(self._get_inventory(agent)),
            "adjacent_tiles": self._adjacent_tiles_percepts(agent),
            "action_success": action_success,
            "disposed_counts": dict(self.disposed_counts),
        }
