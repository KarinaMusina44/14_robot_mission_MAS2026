"""
Group: 14
Date: 16 March 2026
Members: Deodato V. Bastos Neto, Karina Musina
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
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
        vision: int = 1,
        green_coordination: bool = False,
        log_communications: bool = False,
        use_memory: bool = True,
        patrol_border: bool = False,
        use_communication: bool = True,
        multiple_wastes: bool = False,
        rng: Optional[Union[int, np.random.Generator]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if rng is None and seed is not None:
            rng = seed
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        super().__init__(rng=rng)
        self.width = width
        self.height = height
        self.green_coordination = bool(green_coordination)
        self.log_communications = bool(log_communications)
        self.grid: MultiGrid = MultiGrid(width, height, torus=False)
        self.running = True

        self.vision = vision
        self.use_memory = use_memory
        self.patrol_border = patrol_border
        self.use_communication = use_communication
        self.multiple_wastes = multiple_wastes

        self.radioactivity_agents: List[Radioactivity] = []
        self.waste_agents: List[Waste] = []
        self.robot_agents: List[RobotAgent] = []
        self.waste_disposal_zone: Optional[WasteDisposalZone] = None
        self.waste_disposal_pos: Optional[Position] = None
        self.disposed_counts: Dict[str, int] = {"green": 0, "yellow": 0, "red": 0}
        self.cumulative_moves: int = 0
        self.cumulative_moves_green: int = 0
        self.cumulative_moves_yellow: int = 0
        self.cumulative_moves_red: int = 0
        self.time_to_clear = None
        self.communication_events: List[str] = []

        self._init_radioactivity_field()
        self._init_waste_disposal_zone()
        self._init_waste(n_waste)
        self._init_robots(
            n_robots=n_robots,
            n_green_robots=n_green_robots,
            n_yellow_robots=n_yellow_robots,
            n_red_robots=n_red_robots,
        )
        self._init_datacollector()
        self.datacollector.collect(self)

    def _init_datacollector(self) -> None:
        self.datacollector = DataCollector(
            model_reporters={
                "time": "time",
                "waste_total": self._report_waste_total,
                "waste_green": self._report_waste_green,
                "waste_yellow": self._report_waste_yellow,
                "waste_red": self._report_waste_red,
                "inventory_total": self._report_inventory_total,
                "system_waste_total": self._report_system_waste_total,
                "cumulative_moves": "cumulative_moves",
                "cumulative_moves_green": "cumulative_moves_green",
                "cumulative_moves_yellow": "cumulative_moves_yellow",
                "cumulative_moves_red": "cumulative_moves_red",
                "time_to_clear": "time_to_clear",
            }
        )

    def _format_comm_payload(self, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        compact: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, list) and len(value) > 8:
                compact[key] = f"list(len={len(value)})"
            else:
                compact[key] = value
        return compact

    def log_communication_event(
        self,
        sender: RobotAgent,
        message: Dict[str, Any],
        recipients: List[RobotAgent],
    ) -> None:
        if not bool(getattr(self, "log_communications", False)):
            return

        sender_label = f"{getattr(sender, 'type', sender.__class__.__name__)}#{getattr(sender, 'unique_id', '?')}"
        recipient_labels = [
            f"{getattr(r, 'type', r.__class__.__name__)}#{getattr(r, 'unique_id', '?')}"
            for r in recipients
        ]
        topic = message.get("topic", "unknown")
        data = self._format_comm_payload(message.get("data"))
        line = (
            f"[COMM t={int(getattr(self, 'time', 0))}] "
            f"{sender_label} -> {recipient_labels} | topic={topic} | data={data}"
        )
        self.communication_events.append(line)
        if len(self.communication_events) > 500:
            self.communication_events = self.communication_events[-500:]
        print(line, flush=True)

    def _waste_counts_total(self) -> Dict[str, int]:
        counts = {"green": 0, "yellow": 0, "red": 0}
        for waste in self.waste_agents:
            waste_type = getattr(waste, "waste_type", None)
            if waste_type in counts:
                counts[waste_type] += 1
        return counts

    def _report_waste_total(self) -> int:
        counts = self._waste_counts_total()
        return counts["green"] + counts["yellow"] + counts["red"]

    def _report_waste_green(self) -> int:
        return self._waste_counts_total()["green"]

    def _report_waste_yellow(self) -> int:
        return self._waste_counts_total()["yellow"]

    def _report_waste_red(self) -> int:
        return self._waste_counts_total()["red"]

    def _report_inventory_total(self) -> int:
        total = 0
        for robot in self.robot_agents:
            inventory = self._get_inventory(robot)
            total += int(inventory.get("green", 0) or 0)
            total += int(inventory.get("yellow", 0) or 0)
            total += int(inventory.get("red", 0) or 0)
        return total

    def _report_system_waste_total(self) -> int:
        return self._report_waste_total() + self._report_inventory_total()

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
        # Distribute wastes safely to avoid mathematical deadlocks:
        # Green must be a multiple of 4, Yellow a multiple of 2.
        if self.multiple_wastes:
            n_green = max(4, int(n_waste * 0.6) // 4 * 4)
            n_yellow = max(2, int(n_waste * 0.3) // 2 * 2)
            n_red = max(1, n_waste - n_green - n_yellow)
        else:
            n_green = max(4, n_waste // 4 * 4)
            n_yellow = 0
            n_red = 0

        z1_max = max(1, int(self.width / 3.0))
        z2_max = max(z1_max + 1, int(self.width * 2 / 3.0))
        z3_max = max(z2_max + 1, self.width - 1)

        # Spawn Green (many, in Z1)
        for _ in range(n_green):
            x = int(self.rng.integers(0, z1_max))
            y = int(self.rng.integers(0, self.height))
            self.add_one_waste_at((x, y), "green")

        # Spawn Yellow (some, in Z2)
        for _ in range(n_yellow):
            x = int(self.rng.integers(z1_max, z2_max))
            y = int(self.rng.integers(0, self.height))
            self.add_one_waste_at((x, y), "yellow")

        # Spawn Red (a few, in Z3)
        for _ in range(n_red):
            x = int(self.rng.integers(z2_max, z3_max))
            y = int(self.rng.integers(0, self.height))
            self.add_one_waste_at((x, y), "red")


    def _init_robots(
        self,
        n_robots: int,
        n_green_robots: int,
        n_yellow_robots: int,
        n_red_robots: int,
    ) -> None:
        if n_robots > 0 and (n_green_robots + n_yellow_robots + n_red_robots) == 0:
            for _ in range(n_robots):
                robot_cls = self.random.choice(
                    [GreenAgent, YellowAgent, RedAgent])
                self._spawn_one_robot(robot_cls)
            return

        for _ in range(max(0, n_green_robots)):
            self._spawn_one_robot(GreenAgent)
        for _ in range(max(0, n_yellow_robots)):
            self._spawn_one_robot(YellowAgent)
        for _ in range(max(0, n_red_robots)):
            self._spawn_one_robot(RedAgent)

    def _spawn_one_robot(
        self,
        robot_cls: type[RobotAgent],
    ) -> None:
        robot = robot_cls(
            model=self,
            vision=self.vision,
            green_coordination=self.green_coordination,
            use_memory=self.use_memory,
            patrol_border=self.patrol_border,
            use_communication=self.use_communication,
        )
        pos = self._random_position_in_zones(
            robot.allowed_zones, avoid_robot_occupied=True
        )
        self.grid.place_agent(robot, pos)
        self.robot_agents.append(robot)

    def _random_position_in_zones(
        self, zones: set[str], avoid_robot_occupied: bool = False
    ) -> Position:
        candidates: List[Position] = []
        for x in range(self.width):
            if self._zone_for_x(x) not in zones:
                continue
            for y in range(self.height):
                if avoid_robot_occupied and self._has_robot_at((x, y)):
                    continue
                candidates.append((x, y))
        if candidates:
            return self.random.choice(candidates)
        if avoid_robot_occupied:
            return self._random_position_in_zones(zones, avoid_robot_occupied=False)
        return (0, 0)

    def step(self) -> None:
        # Keep per-agent toggle in sync so UI parameter changes can apply live.
        enabled = bool(getattr(self, "green_coordination", False))
        for robot in self.robot_agents:
            knowledge = getattr(robot, "knowledge", None)
            if isinstance(knowledge, dict):
                knowledge["green_coordination"] = enabled

        robots = list(self.robot_agents)
        self.random.shuffle(robots)
        for robot in robots:
            robot.step()

        if self.time_to_clear is None and self._report_system_waste_total() <= 0:
            self.time_to_clear = float(getattr(self, "time", 0.0))
            self.running = False

        self.datacollector.collect(self)

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
                    self.grid.move_agent(
                        agent, self.random.choice(fallback_moves))
                    action_success = True

        elif action_type == "move_west":
            west_moves = self._west_moves_for(agent)
            if west_moves:
                self.grid.move_agent(agent, self.random.choice(west_moves))
                action_success = True
            else:
                fallback_moves = self._allowed_moves_for(agent)
                if fallback_moves:
                    self.grid.move_agent(agent, self.random.choice(fallback_moves))
                    action_success = True

        elif action_type == "move_vertical":
            vertical_moves = self._vertical_moves_for(agent)
            if vertical_moves:
                self.grid.move_agent(agent, self.random.choice(vertical_moves))
                action_success = True
            else:
                fallback_moves = self._allowed_moves_for(agent)
                if fallback_moves:
                    self.grid.move_agent(
                        agent, self.random.choice(fallback_moves))
                    action_success = True

        elif action_type == "pickup":
            waste_type = self._action_get(action, "waste")
            if waste_type in {"green", "yellow", "red"} and self.remove_one_waste_at(
                agent.pos, waste_type # type: ignore
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

        elif action_type == "transfer_green":
            receiver_id = self._action_get(action, "to_id")
            count = int(self._action_get(action, "count", 1) or 0)
            receiver = self._robot_by_unique_id(receiver_id)
            if (
                isinstance(agent, GreenAgent)
                and isinstance(receiver, GreenAgent)
                and receiver is not agent
                and count > 0
                and inventory.get("green", 0) >= count
                and self._can_transfer_between(agent, receiver)
            ):
                receiver_inventory = self._get_inventory(receiver)
                inventory["green"] -= count
                receiver_inventory["green"] += count
                action_success = True
                self.log_communication_event(
                    sender=agent,
                    message={
                        "topic": "transfer_green",
                        "data": {"count": count},
                    },
                    recipients=[receiver],
                )

        elif action_type == "drop":
            waste_type = self._action_get(action, "waste")
            if waste_type in inventory and inventory[waste_type] > 0:
                inventory[waste_type] -= 1
                self.add_one_waste_at(agent.pos, waste_type) # type: ignore
                action_success = True

        elif action_type == "put_away":
            waste_type = self._action_get(action, "waste")
            if (
                waste_type in inventory
                and inventory[waste_type] > 0
                and self._is_disposal_cell(agent.pos) # type: ignore
            ):
                inventory[waste_type] -= 1
                self.disposed_counts[waste_type] += 1
                action_success = True

        elif action_type == "wait":
            action_success = True

        if action_success and action_type in {
            "move",
            "move_random",
            "move_east",
            "move_west",
            "move_vertical",
        }:
            self.cumulative_moves += 1
            if isinstance(agent, GreenAgent):
                self.cumulative_moves_green += 1
            elif isinstance(agent, YellowAgent):
                self.cumulative_moves_yellow += 1
            elif isinstance(agent, RedAgent):
                self.cumulative_moves_red += 1

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
        neighbors = self.grid.get_neighborhood(
            agent.pos, moore=False, include_center=False) # type: ignore
        return [
            p
            for p in neighbors
            if self._zone_for_x(p[0]) in agent.allowed_zones and not self._has_robot_at(p)
        ]

    def _east_moves_for(self, agent: RobotAgent) -> List[Position]:
        return [p for p in self._allowed_moves_for(agent) if p[0] > agent.pos[0]] # type: ignore

    def _west_moves_for(self, agent: RobotAgent) -> List[Position]:
        return [p for p in self._allowed_moves_for(agent) if p[0] < agent.pos[0]] # type: ignore

    def _vertical_moves_for(self, agent: RobotAgent) -> List[Position]:
        return [p for p in self._allowed_moves_for(agent) if p[0] == agent.pos[0]] # type: ignore

    def _is_move_feasible(self, agent: RobotAgent, target: Position) -> bool:
        return target in self._allowed_moves_for(agent)

    def _has_robot_at(self, pos: Position, exclude: Optional[RobotAgent] = None) -> bool:
        for obj in self.grid.get_cell_list_contents([pos]):
            if isinstance(obj, RobotAgent) and obj is not exclude:
                return True
        return False

    def _is_drop_feasible(self, agent: RobotAgent) -> bool:
        target_zone = getattr(agent, "next_zone_for_drop", None)
        if not isinstance(target_zone, str):
            return False
        for p in self.grid.get_neighborhood(agent.pos, moore=False, include_center=False): # type: ignore
            if p[0] > agent.pos[0] and self._zone_for_x(p[0]) == target_zone: # type: ignore
                return True
        return False

    def _robot_by_unique_id(self, unique_id: Any) -> Optional[RobotAgent]:
        for robot in self.robot_agents:
            if getattr(robot, "unique_id", None) == unique_id:
                return robot
        return None

    def _can_transfer_between(self, sender: RobotAgent, receiver: RobotAgent) -> bool:
        sx, sy = sender.pos # type: ignore
        rx, ry = receiver.pos # type: ignore
        return max(abs(sx - rx), abs(sy - ry)) <= 1

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
        cells = self.grid.get_neighborhood(
            agent.pos, moore=True, include_center=True) # type: ignore
        for pos in cells:
            contents = self.grid.get_cell_list_contents([pos])
            data[pos] = {
                "zone": self._zone_for_x(pos[0]),
                "is_disposal_zone": self._is_disposal_cell(pos),
                "wastes": self._cell_wastes(pos),
                "contents": [obj.__class__.__name__ for obj in contents],
            }
        return data

    def _visible_tiles_percepts(self, agent: RobotAgent) -> Dict[Position, Dict[str, Any]]:
        data: Dict[Position, Dict[str, Any]] = {}
        radius = max(1, int(getattr(agent, "vision", 1) or 1))
        cells = self.grid.get_neighborhood(
            agent.pos, moore=True, include_center=False, radius=radius # type: ignore
        )
        for pos in cells:
            data[pos] = {
                "zone": self._zone_for_x(pos[0]),
                "is_disposal_zone": self._is_disposal_cell(pos),
                "wastes": self._cell_wastes(pos),
            }
        return data

    def _build_percepts(self, agent: RobotAgent, action_success: bool) -> Dict[str, Any]:
        next_zone = getattr(agent, "next_zone_for_drop", None)
        frontier_to_next_zone = False
        if isinstance(next_zone, str):
            for pos in self.grid.get_neighborhood(agent.pos, moore=False, include_center=False): # type: ignore
                if pos[0] > agent.pos[0] and self._zone_for_x(pos[0]) == next_zone: # type: ignore
                    frontier_to_next_zone = True
                    break

        return {
            "position": tuple(agent.pos), # type: ignore
            "zone": self._zone_for_x(agent.pos[0]), # type: ignore
            "cell_wastes": self._cell_wastes(agent.pos), # type: ignore
            "allowed_moves": self._allowed_moves_for(agent),
            "in_disposal_zone": self._is_disposal_cell(agent.pos), # type: ignore
            "frontier_to_next_zone_for_drop": frontier_to_next_zone,
            "frontier_to_next_zone": frontier_to_next_zone,
            "visible_tiles": self._visible_tiles_percepts(agent),
            "inventory": dict(self._get_inventory(agent)),
            "adjacent_tiles": self._adjacent_tiles_percepts(agent),
            "action_success": action_success,
            "disposed_counts": dict(self.disposed_counts),
        }
