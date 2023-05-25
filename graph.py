import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from data import Data
import ast

from rdkit import Chem
from rdkit import DataStructs

class Graph:
    def __init__(self):
        self.num_occurences = None # # number of times a metabolite apperas on pairs dataset
        self.G = None # Graph structure

    def get_number_of_occurences(self, pairs):
        self.num_occurences = pd.DataFrame(pd.DataFrame(pd.concat([pairs['source'], pairs['target']], axis=0)).value_counts())
        
    def _get_num_occur(self, a,b):
        t_a = self.num_occurences.loc[a]
        t_b = self.num_occurences.loc[b]
        w = max(t_a.values[0][0], t_b.values[0][0])
        return w

    def _get_mol_weight(self, data: Data, a, b):

        if data.get_compound_by_id(a).is_cofactor or data.get_compound_by_id(b).is_cofactor:
            return 999
        
        w_a = data.get_compound_by_id(a).mw
        w_b = data.get_compound_by_id(b).mw
        w = (np.abs(w_a-w_b) / (w_a+w_b+1e-6))
        return w

    def create_graph(self, data: Data, pairs):
        self.G = nx.from_pandas_edgelist(pairs, source='source', target='target', create_using=nx.Graph()) 
        print('# nodes:', self.G.number_of_nodes(), "\n# edges:", self.G.number_of_edges())

        print('\nRemoving self-loops...')
        self_loops = list(nx.selfloop_edges(self.G))
        self.G.remove_edges_from(self_loops)
        print('# nodes:', self.G.number_of_nodes(), "\n# edges:", self.G.number_of_edges())

        # set node attributes
        for node in self.G.nodes():
            self.G.nodes[node]['mw'] = data.get_compound_by_id(node).mw
        
    def simple_weighted_shortest_path(self, data: Data, test_cases, method):
        '''
        method:
            - num_occur: Weight based on number of occurences of the metabolites of the pairs
        '''
        if method=='num_occur':
            for edge in tqdm(self.G.edges()):
                self.G.edges[(edge[0], edge[1])]['num_occur'] = self._get_num_occur(a=edge[0], b=edge[1])
        elif method=='mol_weight':
            for edge in tqdm(self.G.edges()):
                self.G.edges[(edge[0], edge[1])]['mol_weight'] = self._get_mol_weight(data=data, a=edge[0], b=edge[1])

        correct_pathways = []
        paths = []
        for row in range(len(test_cases)):
            source = test_cases['source'].iloc[row]
            target = test_cases['target'].iloc[row]
            correct_pathways.append((list(nx.shortest_path(self.G, source, target, weight=method)) == test_cases['paths_list'].iloc[row]))
            paths.append(list(nx.shortest_path(self.G, source, target, weight=method)))

        print(f'Correct pathway predictions: {correct_pathways.count(True)}')
        print(f'Correct pathway predictions (%): {100 * correct_pathways.count(True) / len(correct_pathways)}')

        # return the DataFrame with the resulted pathways and correct or not
        paths = pd.DataFrame([str(p) for p in paths], columns=['Pathway'])
        paths['Pathway']  = paths['Pathway'].apply(lambda x: ast.literal_eval(x))
        paths['Correct'] = correct_pathways
        return paths
    
