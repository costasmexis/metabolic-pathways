import pandas as pd
import numpy as np
import json

class Reaction:
    def __init__(self, entry, name, compounds, enzyme):
        self.entry = entry
        self.name = name
        self.compounds = json.loads(compounds)
        self.enzyme = enzyme
    
    def __str__(self):
        return f"ID: {self.entry}\nName: {self.name}\nCompounds: {self.compounds}\n"

