import pandas as pd
import numpy as np
import networkx as nx

class Compound:
    def __init__(self, entry, name, formula, mw, smiles):
        self.entry = entry
        self.name = name
        self.formula = formula
        self.mw = mw
        self.smiles = smiles

    def __str__(self):
        return f"ID: {self.entry}\nName: {self.name}\nFormula: {self.formula}\n"
