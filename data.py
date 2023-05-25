import pandas as pd
import numpy as np
import json

class Data:
    def __init__(self):
        self.compounds = {}   # Compounds
        self.reactions = {}   # Reactions

    def add_element(self, element_type, element):
        element_dict = {'compound': self.compounds, 'reaction': self.reactions}
        if element_type in element_dict:
            element_dict[element_type][element.entry] = element
        else:
            return 'Wrong type...'

    def get_compound_by_id(self, entry):
        return self.compounds.get(entry)

    def print_all_compounds(self):
        for compound in self.compounds.values():
            print(compound)

    def get_reaction_by_id(self, entry):
        return self.reactions.get(entry)
    
    def print_all_reactions(self):
        for reaction in self.reactions.values():
            print(reaction)

