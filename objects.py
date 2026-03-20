"""
Group: 14
Date: 16 March 2026
Members: Deodato V. Bastos Neto, Karina Musina
"""

import random

from mesa import Agent


class Radioactivity(Agent):
    def __init__(self, model, zone):
        super().__init__(model)
        self.zone = zone

        if zone == 'z3':
            self.radioactivity = random.uniform(0.66, 1.0)
        elif zone == 'z2':
            self.radioactivity = random.uniform(0.33, 0.66)
        elif zone == 'z1':
            self.radioactivity = random.uniform(0.0, 0.33)
        else:
            raise ValueError("Invalid zone. Choose 'z1', 'z2', or 'z3'.")


class WasteDisposalZone(Agent):
    def __init__(self, model, zone):
        super().__init__(model)
        self.zone = zone


class Waste(Agent):
    def __init__(self, model, waste_type):
        super().__init__(model)
        self.waste_type = waste_type