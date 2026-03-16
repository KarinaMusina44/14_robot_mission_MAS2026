from mesa import Agent


class robotAgent(Agent):
    """Simple shared base class for the three robots."""

    allowed_zones = {"z1"}
    next_zone_for_drop = None
    robot_color = "unknown"

    def __init__(self, model):
        super().__init__(model)
        self.type = self.__class__.robot_color
        self.allowed_zones = set(self.__class__.allowed_zones)
        self.knowledge = {
            "inventory": {"green": 0, "yellow": 0, "red": 0},
            "last_percepts": {},
            "last_action": None,
            "history": [],
            "model_percepts": {},
        }

    def zone_of_cell(self, pos):
        """Read zone from an object placed on the cell."""
        contents = self.model.grid.get_cell_list_contents([pos])

        for obj in contents:
            zone_attr = getattr(obj, "zone", None)
            if zone_attr in {"z1", "z2", "z3"}:
                return zone_attr

        return None

    def in_disposal_zone(self, pos):
        """Detect disposal cell from an object's `zone` attribute."""
        contents = self.model.grid.get_cell_list_contents([pos])
        for obj in contents:
            zone_attr = getattr(obj, "zone", None)
            if isinstance(zone_attr, str) and "disposal" in zone_attr.lower():
                return True
        return False

    def allowed_moves(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )

        possible = []
        for p in neighbors:
            zone = self.zone_of_cell(p)
            if zone in self.allowed_zones:
                possible.append(p)
        return possible

    def has_east_neighbor_in_zone(self, target_zone):
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        x = self.pos[0]
        for p in neighbors:
            if p[0] > x and self.zone_of_cell(p) == target_zone:
                return True
        return False

    def cell_wastes(self):
        """Count waste objects on the current cell."""
        counts = {"green": 0, "yellow": 0, "red": 0}
        contents = self.model.grid.get_cell_list_contents([self.pos])

        for obj in contents:
            if obj is self:
                continue

            waste = getattr(obj, "waste_type", None)
            if waste in counts:
                counts[waste] += 1
                continue

            # Fallback if waste is identified only by class name or color.
            label = f"{getattr(obj, 'color', '')} {obj.__class__.__name__}".lower()
            if "green" in label:
                counts["green"] += 1
            elif "yellow" in label:
                counts["yellow"] += 1
            elif "red" in label:
                counts["red"] += 1

        return counts

    def percepts(self):
        frontier_to_next_zone = False
        if self.next_zone_for_drop is not None:
            frontier_to_next_zone = self.has_east_neighbor_in_zone(self.next_zone_for_drop)

        return {
            "position": self.pos,
            "zone": self.zone_of_cell(self.pos),
            "cell_wastes": self.cell_wastes(),
            "allowed_moves": self.allowed_moves(),
            "in_disposal_zone": self.in_disposal_zone(self.pos),
            "frontier_to_next_zone": frontier_to_next_zone,
        }

    def move_random(self, possible_moves):
        if not possible_moves:
            return
        new_position = self.random.choice(possible_moves)
        self.model.grid.move_agent(self, new_position)
        self.pos = new_position

    def apply_action(self, action):
        """Local action execution when model.do is not implemented."""
        inv = self.knowledge["inventory"]
        action_type = action.get("type")

        if action_type == "move_random":
            self.move_random(self.allowed_moves())

        elif action_type == "move_east":
            moves = self.allowed_moves()
            x = self.pos[0]
            east_moves = [m for m in moves if m[0] > x]
            if east_moves:
                self.move_random(east_moves)
            else:
                self.move_random(moves)

        elif action_type == "pickup":
            waste = action["waste"]
            inv[waste] += 1
            if hasattr(self.model, "remove_one_waste_at"):
                self.model.remove_one_waste_at(self.pos, waste)

        elif action_type == "transform":
            src = action["from"]
            dst = action["to"]
            count = action["count"]
            if inv[src] >= count:
                inv[src] -= count
                inv[dst] += 1

        elif action_type == "put_away":
            waste = action["waste"]
            if inv[waste] > 0:
                inv[waste] -= 1

        elif action_type == "drop":
            waste = action["waste"]
            if inv[waste] > 0:
                inv[waste] -= 1
                if hasattr(self.model, "add_one_waste_at"):
                    self.model.add_one_waste_at(self.pos, waste)
                elif hasattr(self.model, "add_waste_at"):
                    self.model.add_waste_at(self.pos, waste)

    def do(self, action):
        if hasattr(self.model, "do"):
            return self.model.do(self, action)

        self.apply_action(action)
        return self.percepts()

    def update_knowledge(self, percepts, action=None, model_percepts=None):
        self.knowledge["last_percepts"] = percepts
        if action is not None:
            self.knowledge["last_action"] = action
        if isinstance(model_percepts, dict):
            self.knowledge["model_percepts"] = model_percepts
        self.knowledge["history"].append(
            {
                "percepts": percepts,
                "action": action,
                "model_percepts": model_percepts if isinstance(model_percepts, dict) else {},
            }
        )

    def step(self):
        self.step_agent()

    def step_agent(self):
        percepts = self.percepts()
        self.update_knowledge(percepts)
        action = self.deliberate(self.knowledge)
        model_percepts = self.do(action)
        new_percepts = self.percepts()
        self.update_knowledge(new_percepts, action=action, model_percepts=model_percepts)


class greenAgent(robotAgent):
    """z1 only, pick green, transform 2 green -> 1 yellow, then carry east."""

    allowed_zones = {"z1"}
    next_zone_for_drop = "z2"
    robot_color = "green"

    @staticmethod
    def deliberate(knowledge):
        p = knowledge["last_percepts"]
        inv = knowledge["inventory"]

        if inv["green"] >= 2:
            return {"type": "transform", "from": "green", "to": "yellow", "count": 2}

        if inv["yellow"] >= 1:
            if p["frontier_to_next_zone"]:
                return {"type": "drop", "waste": "yellow"}
            return {"type": "move_east"}

        if p["cell_wastes"]["green"] > 0:
            return {"type": "pickup", "waste": "green"}

        if p["allowed_moves"]:
            return {"type": "move_random"}
        return {"type": "wait"}


class yellowAgent(robotAgent):
    """z1-z2, pick yellow, transform 2 yellow -> 1 red, then carry east."""

    allowed_zones = {"z1", "z2"}
    next_zone_for_drop = "z3"
    robot_color = "yellow"

    @staticmethod
    def deliberate(knowledge):
        p = knowledge["last_percepts"]
        inv = knowledge["inventory"]

        if inv["yellow"] >= 2:
            return {"type": "transform", "from": "yellow", "to": "red", "count": 2}

        if inv["red"] >= 1:
            if p["frontier_to_next_zone"]:
                return {"type": "drop", "waste": "red"}
            return {"type": "move_east"}

        if p["cell_wastes"]["yellow"] > 0:
            return {"type": "pickup", "waste": "yellow"}

        if p["allowed_moves"]:
            return {"type": "move_random"}
        return {"type": "wait"}


class redAgent(robotAgent):
    """z1-z2-z3, pick red, carry east, and put away in disposal zone."""

    allowed_zones = {"z1", "z2", "z3"}
    robot_color = "red"

    @staticmethod
    def deliberate(knowledge):
        p = knowledge["last_percepts"]
        inv = knowledge["inventory"]

        if inv["red"] >= 1:
            if p["in_disposal_zone"]:
                return {"type": "put_away", "waste": "red"}
            return {"type": "move_east"}

        if p["cell_wastes"]["red"] > 0:
            return {"type": "pickup", "waste": "red"}

        if p["allowed_moves"]:
            return {"type": "move_random"}
        return {"type": "wait"}
