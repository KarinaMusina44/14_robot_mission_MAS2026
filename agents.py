"""
Group: 14
Date: 16 March 2026
Members: Deodato V. Bastos Neto, Karina Musina
"""

import random
from matplotlib.pylab import inv
from mesa import Agent

class RobotAgent(Agent):
    """Simple shared base class for the three robots."""

    allowed_zones = {"z1"}
    next_zone_for_drop = None
    prev_zone = None
    robot_color = "unknown"

    def __init__(self, model, vision=2, use_memory=True, patrol_border=True, use_communication=True, log_messages=True):
        super().__init__(model)
        self.vision = vision
        self.type = self.__class__.robot_color
        self.allowed_zones = set(self.__class__.allowed_zones)
        self.knowledge = {
            "inventory": {"green": 0, "yellow": 0, "red": 0},
            "last_percepts": {},
            "last_action": None,
            "history": [],
            "model_percepts": {},
            "use_memory": use_memory,
            "patrol_border": patrol_border,
            "inbox": [],
            "use_communication": use_communication,
            "log_messages": log_messages,
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
        if hasattr(self.model, "_allowed_moves_for"):
            model_moves = self.model._allowed_moves_for(self)
            if isinstance(model_moves, list):
                return model_moves

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
        x = self.pos[0] # type: ignore
        for p in neighbors:
            if p[0] > x and self.zone_of_cell(p) == target_zone:
                return True
        return False

    def has_west_neighbor_in_zone(self, target_zone):
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        x = self.pos[0] # type: ignore
        for p in neighbors:
            if p[0] < x and self.zone_of_cell(p) == target_zone:
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

        return counts

    def get_visible_tiles(self):
        """Build percepts for all tiles within the vision radius."""
        visible = {}
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False, radius=self.vision
        )
        for pos in neighbors:
            contents = self.model.grid.get_cell_list_contents([pos])
            
            zone_attr = None
            is_disp = False
            wastes = {"green": 0, "yellow": 0, "red": 0}
            
            for obj in contents:
                z = getattr(obj, "zone", None)
                if z in {"z1", "z2", "z3"}:
                    zone_attr = z
                if isinstance(z, str) and "disposal" in z.lower():
                    is_disp = True
                    
                w = getattr(obj, "waste_type", None)
                if w in wastes:
                    wastes[w] += 1

            visible[pos] = {
                "zone": zone_attr,
                "is_disposal_zone": is_disp,
                "wastes": wastes
            }
        return visible

    def percepts(self):
        frontier_to_next_zone_for_drop = False
        if self.next_zone_for_drop is not None:
            frontier_to_next_zone_for_drop = self.has_east_neighbor_in_zone(self.next_zone_for_drop)

        return {
            "id": self.unique_id,
            "position": self.pos,
            "zone": self.zone_of_cell(self.pos),
            "cell_wastes": self.cell_wastes(),
            "allowed_moves": self.allowed_moves(),
            "in_disposal_zone": self.in_disposal_zone(self.pos),
            "frontier_to_next_zone_for_drop": frontier_to_next_zone_for_drop,
            "visible_tiles": self.get_visible_tiles(), # Expose extended vision
        }

    @staticmethod
    def best_move_towards(target, allowed_moves):
        """Helper to step towards a distant target."""
        if target in allowed_moves:
            return target

        best_moves = []
        best_dist = float('inf')
        for m in allowed_moves:
            dist = abs(m[0] - target[0]) + abs(m[1] - target[1])
            if dist < best_dist:
                best_dist = dist
                best_moves = [m]
            elif dist == best_dist:
                best_moves.append(m)

        # Pick randomly among equal-distance best moves to prevent oscillating
        if best_moves:
            return random.choice(best_moves)
        return None

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
            x = self.pos[0] # type: ignore
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
        if isinstance(percepts, dict) and isinstance(percepts.get("inventory"), dict):
            self.knowledge["inventory"].update(percepts["inventory"])
        if isinstance(model_percepts, dict):
            self.knowledge["model_percepts"] = model_percepts
            if isinstance(model_percepts.get("inventory"), dict):
                self.knowledge["inventory"].update(model_percepts["inventory"])
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

        if "messages" in action and self.knowledge.get("use_communication", True):
            for msg in action["messages"]:
                for other_agent in self.model.robot_agents:
                    if other_agent is not self and other_agent.type == msg["to"]:
                        other_agent.knowledge["inbox"].append(msg)

                        if self.knowledge.get("log_messages", True):
                            step = getattr(self.model, "steps", "?")
                            print(f"[Step {step}] {self.type.capitalize()} Agent {self.unique_id} -> {other_agent.type.capitalize()} Agent {other_agent.unique_id} | Topic: '{msg['topic']}' | Data: {msg['data']}")

        model_percepts = self.do(action)
        new_percepts = self.percepts()
        self.update_knowledge(new_percepts, action=action, model_percepts=model_percepts)

    @staticmethod
    def deliberate(knowledge):
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement deliberate() method.")


class GreenAgent(RobotAgent):
    """z1 only, pick green, transform 2 green -> 1 yellow, then carry east."""

    allowed_zones = {"z1"}
    next_zone_for_drop = "z2"
    robot_color = "green"

    @staticmethod
    def deliberate(knowledge):
        p = knowledge["last_percepts"]
        inv = knowledge["inventory"]
        vis = p.get("visible_tiles", {})

        # Negotiate Deadlocks & Find Wastes
        knowledge.setdefault("known_wastes", set())
        knowledge.setdefault("yielded_wastes", set())
        yield_to_peer = False

        for msg in knowledge.get("inbox", []):
            if msg["topic"] == "dropped_waste":
                knowledge["known_wastes"].add(msg["data"])
            elif msg["topic"] == "holding_one":
                # Negotiation: If peer also has 1, the lower ID yields (drops theirs)
                peer_id = msg["data"]["id"]
                if inv["green"] == 1 and p["id"] < peer_id:
                    yield_to_peer = True
        knowledge["inbox"] = []

        # Memory pruning: Remove green wastes from memory if they are no longer there
        to_remove = set()
        for kw in knowledge["known_wastes"]:
            if kw == p["position"] and p["cell_wastes"]["green"] == 0:
                to_remove.add(kw)
        knowledge["known_wastes"] -= to_remove

        to_remove_yielded = set()
        for yw in knowledge["yielded_wastes"]:
            if (yw == p["position"] and p["cell_wastes"]["green"] == 0) or \
               (yw in vis and vis[yw]["wastes"]["green"] == 0):
                to_remove_yielded.add(yw)
        knowledge["yielded_wastes"] -= to_remove_yielded

        messages_to_send = []
        def with_msgs(action_dict):
            if messages_to_send:
                action_dict["messages"] = action_dict.get("messages", []) + messages_to_send
            return action_dict

        # 1. Check if we can transform
        if inv["green"] >= 2:
            return {"type": "transform", "from": "green", "to": "yellow", "count": 2}

        # 2. Check if we have yellow waste to move/drop
        if inv["yellow"] >= 1:
            if p["frontier_to_next_zone_for_drop"]:
                messages_to_send.append({"to": "yellow", "topic": "dropped_waste", "data": p["position"]})
                return with_msgs({"type": "drop", "waste": "yellow"})
            return with_msgs({"type": "move_east"})

        # 3. Negotiation: If we are holding one and see another, yield to the peer with the lower ID
        if inv["green"] == 1:
            if not knowledge.get("use_communication", True):
                knowledge["frustration"] = knowledge.get("frustration", 0) + 1
                if knowledge["frustration"] > 20:
                    knowledge["frustration"] = 0
                    return with_msgs({"type": "drop", "waste": "green"})
            else:
                if yield_to_peer:
                    messages_to_send.append({"to": "green", "topic": "dropped_waste", "data": p["position"]})
                    knowledge["frustration"] = 0
                    knowledge["yielded_wastes"].add(p["position"])
                    return with_msgs({"type": "drop", "waste": "green"})
                elif not knowledge.get("known_wastes"):
                    knowledge["frustration"] = knowledge.get("frustration", 0) + 1
                    if knowledge["frustration"] % 10 == 0:
                        messages_to_send.append({"to": "green", "topic": "holding_one", "data": {"id": p["id"], "pos": p["position"]}})
        else:
            knowledge["frustration"] = 0

        # 4. Check if there is green waste on the current tile
        if p["cell_wastes"]["green"] > 0 and p["position"] not in knowledge["yielded_wastes"]:
            return with_msgs({"type": "pickup", "waste": "green"})

        # 5. Pathfinding (Extended Vision)
        targets = [pos for pos, info in vis.items() if info["wastes"]["green"] > 0 and pos not in knowledge["yielded_wastes"]]
        if targets:
            targets.sort(key=lambda t: abs(t[0] - p["position"][0]) + abs(t[1] - p["position"][1]))
            best_move = RobotAgent.best_move_towards(targets[0], p["allowed_moves"])
            if best_move:
                return with_msgs({"type": "move", "to": best_move})

        # 6. Communication Pathfinding
        if knowledge["known_wastes"]:
            targets = list(knowledge["known_wastes"])
            targets.sort(key=lambda t: abs(t[0] - p["position"][0]) + abs(t[1] - p["position"][1]))
            best_move = RobotAgent.best_move_towards(targets[0], p["allowed_moves"])
            if best_move:
                return with_msgs({"type": "move", "to": best_move})

        # 7. Move randomly if nothing else to do
        if p["allowed_moves"]:
            return with_msgs({"type": "move_random"})
        return with_msgs({"type": "wait"})


class YellowAgent(RobotAgent):
    """z1-z2, pick yellow, transform 2 yellow -> 1 red, then carry east."""

    allowed_zones = {"z1", "z2"}
    next_zone_for_drop = "z3"
    robot_color = "yellow"

    @staticmethod
    def deliberate(knowledge):
        p = knowledge["last_percepts"]
        inv = knowledge["inventory"]
        vis = p.get("visible_tiles", {})

         # Read messages from Green robots)
        knowledge.setdefault("known_wastes", set())
        knowledge.setdefault("yielded_wastes", set())
        yield_to_peer = False

        for msg in knowledge.get("inbox", []):
            if msg["topic"] == "dropped_waste":
                knowledge["known_wastes"].add(msg["data"])
            elif msg["topic"] == "holding_one":
                # Negotiation: If peer also has 1, the lower ID yields (drops theirs)
                peer_id = msg["data"]["id"]
                if inv["yellow"] == 1 and p["id"] < peer_id:
                    yield_to_peer = True
        knowledge["inbox"] = []

        # Remove yellow wastes from memory if they are no longer there
        to_remove = set()
        for kw in knowledge["known_wastes"]:
            if kw == p["position"] and p["cell_wastes"]["yellow"] == 0:
                to_remove.add(kw)
        knowledge["known_wastes"] -= to_remove

        to_remove_yielded = set()
        for yw in knowledge["yielded_wastes"]:
            if (yw == p["position"] and p["cell_wastes"]["yellow"] == 0) or \
               (yw in vis and vis[yw]["wastes"]["yellow"] == 0):
                to_remove_yielded.add(yw)
        knowledge["yielded_wastes"] -= to_remove_yielded

        messages_to_send = []
        def with_msgs(action_dict):
            if messages_to_send:
                action_dict["messages"] = action_dict.get("messages", []) + messages_to_send
            return action_dict

        # 1. Check if we can transform
        if inv["yellow"] >= 2:
            return with_msgs({"type": "transform", "from": "yellow", "to": "red", "count": 2})

        # 2. Check if we have red waste to move/drop
        if inv["red"] >= 1:
            if p["frontier_to_next_zone_for_drop"]:
                messages_to_send.append({"to": "red", "topic": "dropped_waste", "data": p["position"]})
                return with_msgs({"type": "drop", "waste": "red"})
            return with_msgs({"type": "move_east"})

        # 3. Handle holding exactly 1 yellow waste
        if inv["yellow"] == 1:
            if not knowledge.get("use_communication", True):
                knowledge["frustration"] = knowledge.get("frustration", 0) + 1
                if knowledge["frustration"] > 20:
                    knowledge["frustration"] = 0
                    return with_msgs({"type": "drop", "waste": "yellow"})
            else:
                if yield_to_peer:
                    messages_to_send.append({"to": "yellow", "topic": "dropped_waste", "data": p["position"]})
                    knowledge["frustration"] = 0
                    knowledge["yielded_wastes"].add(p["position"])
                    return with_msgs({"type": "drop", "waste": "yellow"})
                elif not knowledge.get("known_wastes"):
                    knowledge["frustration"] = knowledge.get("frustration", 0) + 1
                    if knowledge["frustration"] % 10 == 0:
                        messages_to_send.append({"to": "yellow", "topic": "holding_one", "data": {"id": p["id"], "pos": p["position"]}})
        else:
            knowledge["frustration"] = 0

        # 4. Check if there is yellow waste on the CURRENT tile
        if p["cell_wastes"]["yellow"] > 0 and p["position"] not in knowledge["yielded_wastes"]:
            return with_msgs({"type": "pickup", "waste": "yellow"})

        # 5. Pathfinding
        targets = [pos for pos, info in vis.items() if info["wastes"]["yellow"] > 0 and pos not in knowledge["yielded_wastes"]]
        if targets:
            targets.sort(key=lambda t: abs(t[0] - p["position"][0]) + abs(t[1] - p["position"][1]))
            best_move = RobotAgent.best_move_towards(targets[0], p["allowed_moves"])
            if best_move:
                return with_msgs({"type": "move", "to": best_move})

        # 6. Communication Pathfinding
        if knowledge["known_wastes"]:
            targets = list(knowledge["known_wastes"])
            targets.sort(key=lambda t: abs(t[0] - p["position"][0]) + abs(t[1] - p["position"][1]))
            best_move = RobotAgent.best_move_towards(targets[0], p["allowed_moves"])
            if best_move:
                return with_msgs({"type": "move", "to": best_move})

        # 7. Seek border & patrol: Move to the Z1/Z2 border and patrol vertically
        if inv["yellow"] < 2 and knowledge.get("patrol_border", True):
            at_pickup_border = False
            if vis:
                for pos, tile_info in vis.items():
                    if p["zone"] == "z1" and tile_info["zone"] == "z2" and pos[0] == p["position"][0] + 1:
                        at_pickup_border = True

            if at_pickup_border:
                vertical_moves = [m for m in p["allowed_moves"] if m[0] == p["position"][0]]
                if vertical_moves:
                    return with_msgs({"type": "move", "to": random.choice(vertical_moves)})
                return with_msgs({"type": "wait"})

            # If we are not at the border yet, navigate towards it
            if p["zone"] == "z2" and p["allowed_moves"]:
                west_moves = [m for m in p["allowed_moves"] if m[0] < p["position"][0]]
                if west_moves:
                    return with_msgs({"type": "move", "to": random.choice(west_moves)})
            elif p["zone"] == "z1" and p["allowed_moves"]:
                east_moves = [m for m in p["allowed_moves"] if m[0] > p["position"][0]]
                if east_moves:
                    return with_msgs({"type": "move", "to": random.choice(east_moves)})

        # 8. Move randomly if nothing else to do
        if p["allowed_moves"]:
            return with_msgs({"type": "move_random"})

        return with_msgs({"type": "wait"})


class RedAgent(RobotAgent):
    """z1-z2-z3, pick red, carry east, and put away in disposal zone."""

    allowed_zones = {"z1", "z2", "z3"}
    robot_color = "red"

    @staticmethod
    def deliberate(knowledge):
        p = knowledge["last_percepts"]
        inv = knowledge["inventory"]
        adj = knowledge.get("model_percepts", {}).get("adjacent_tiles", {})
        vis = p.get("visible_tiles", {})

        # Read messages from Yellow robots and peers)
        knowledge.setdefault("known_wastes", set())
        for msg in knowledge.get("inbox", []):
            if msg["topic"] == "disposal_zone":
                knowledge["disposal_zone_pos"] = msg["data"]
            elif msg["topic"] == "dropped_waste":
                knowledge["known_wastes"].add(msg["data"])
        knowledge["inbox"] = []

        # Remove red wastes from memory if they are no longer there
        to_remove = set()
        for kw in knowledge["known_wastes"]:
            if kw == p["position"] and p["cell_wastes"]["red"] == 0:
                to_remove.add(kw)
        knowledge["known_wastes"] -= to_remove

        # Save the position of the disposal zone if we see it or are on it
        if knowledge.get("use_memory", True):
            if p["in_disposal_zone"]:
                knowledge["disposal_zone_pos"] = p["position"]
            else:
                for pos, info in vis.items():
                    if info["is_disposal_zone"]:
                        knowledge["disposal_zone_pos"] = pos
                        break

        # Share discovery with peers
        messages_to_send = []
        if knowledge.get("use_communication", True) and "disposal_zone_pos" in knowledge and not knowledge.get("has_broadcasted_disposal"):
            messages_to_send.append({
                "to": "red",
                "topic": "disposal_zone",
                "data": knowledge["disposal_zone_pos"]
            })
            # Mark as broadcasted so we don't spam the network every single turn
            knowledge["has_broadcasted_disposal"] = True

        def with_msgs(action_dict):
            """Helper to attach outgoing messages to the chosen action without consuming a turn."""
            if messages_to_send:
                action_dict["messages"] = messages_to_send
            return action_dict

        # 1. Check if we have red waste to put away
        if inv["red"] >= 1:
            if p["in_disposal_zone"]:
                return with_msgs({"type": "put_away", "waste": "red"})

            # Pathfinding: Move straight to the disposal zone if we remember it
            if knowledge.get("use_memory", True) and "disposal_zone_pos" in knowledge:
                best_move = RobotAgent.best_move_towards(knowledge["disposal_zone_pos"], p["allowed_moves"])
                if best_move:
                    return with_msgs({"type": "move", "to": best_move})

            # Otherwise, keep exploring east until we find it
            return with_msgs({"type": "move_east"})

        # 2. Check if there is red waste on the current tile
        if p["cell_wastes"]["red"] > 0:
            return with_msgs({"type": "pickup", "waste": "red"})

        # 3. Pathfinding: Look for red waste in adjacent tiles
        targets = [pos for pos, info in vis.items() if info["wastes"]["red"] > 0] if vis else []
        if not targets and adj:
            targets = [pos for pos, info in adj.items() if info["wastes"]["red"] > 0]

        if targets:
            targets.sort(key=lambda t: abs(t[0] - p["position"][0]) + abs(t[1] - p["position"][1]))
            best_move = RobotAgent.best_move_towards(targets[0], p["allowed_moves"])
            if best_move:
                return with_msgs({"type": "move", "to": best_move})

        # 4. Communication Pathfinding
        if knowledge["known_wastes"]:
            targets = list(knowledge["known_wastes"])
            targets.sort(key=lambda t: abs(t[0] - p["position"][0]) + abs(t[1] - p["position"][1]))
            best_move = RobotAgent.best_move_towards(targets[0], p["allowed_moves"])
            if best_move:
                return with_msgs({"type": "move", "to": best_move})

        # 5. Seek border & patrol: Move to the Z2/Z3 border and patrol vertically
        if inv["red"] < 1 and knowledge.get("patrol_border", True):
            at_pickup_border = False
            if adj:
                for pos, tile_info in adj.items():
                    # Check if we are at the border looking into the adjacent zone
                    if p["zone"] == "z2" and tile_info["zone"] == "z3" and pos[0] > p["position"][0]:
                        at_pickup_border = True

            if at_pickup_border and p["allowed_moves"]:
                return with_msgs({"type": "move_vertical"})

            # If we are not at the border yet, navigate towards it
            if p["zone"] == "z3" and p["allowed_moves"]:
                return with_msgs({"type": "move_west"})
            elif p["zone"] in ("z1", "z2") and p["allowed_moves"]:
                return with_msgs({"type": "move_east"})

        # 6. Move randomly if nothing else to do
        if p["allowed_moves"]:
            return with_msgs({"type": "move_random"})

        return with_msgs({"type": "wait"})