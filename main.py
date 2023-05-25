import pandas as pd
import numpy as np
import json

import networkx as nx

from tqdm import tqdm

from compound import Compound
from reaction import Reaction
from graph import Graph
from data import Data

# create class instances
data = Data()
graph = Graph()

# read data from csv
cpds = pd.read_csv('data/compounds_final.csv', index_col=0)
rxns = pd.read_csv('data/reactions_final.csv', index_col=0)
pairs = pd.read_csv('data/pairs_final.csv', index_col=0)
cofactors = pd.read_csv('data/original/cofactors_KEGG.csv')

# Create a Compound object for each row in the DataFrame and add it to the data
for index, row in cpds.iterrows():
    entry = row['Entry']
    name = row['Names']
    formula = row['Formula']
    mw = row['mol_weight']
    smiles = row['SMILES']
    is_cofactor = row['Entry'] in cofactors['Entry'].values

    compound = Compound(entry, name, formula, mw, smiles, is_cofactor)
    data.add_element('compound', compound)

# Create a Reaction object for each row in the DataFrame and add it to the data
for index, row in rxns.iterrows():
    entry = row['Entry']
    name = row['Names']
    compounds = row['Compound']
    enzyme = row['EC Number']

    reaction = Reaction(entry, name, compounds, enzyme)
    data.add_element('reaction', reaction)


# number of times a metabolite apperas on pairs dataset
graph.get_number_of_occurences(pairs)

# Create Graph
graph.create_graph(data=data, pairs=pairs)

''' 
*******************************************
Validate the methods on validation datasets 
*******************************************
'''
######### VALIDATION SET FROM nicepath ###########
test_cases = pd.read_csv('data/original/test_cases.csv')
test_cases['source'] = test_cases['Pathway '].apply(lambda x: x.split(',')[0])
test_cases['target'] = test_cases['Pathway '].apply(lambda x: x.split(',')[len(x.split(','))-1])
test_cases['paths_list'] = test_cases['Pathway '].apply(lambda x: x.split(','))

paths = graph.simple_weighted_shortest_path(test_cases=test_cases, data=data, method='mol_weight')

######### NEW VALIDATION SET ###########
pyminer_test = pd.read_csv('data/original/pyminer_validation_set.csv', delimiter=';', header=None, names=['Pathway'])
pyminer_test['source'] = pyminer_test['Pathway'].apply(lambda x: x.split(',')[0])
pyminer_test['target'] = pyminer_test['Pathway'].apply(lambda x: x.split(',')[len(x.split(','))-1])
pyminer_test['paths_list'] = pyminer_test['Pathway'].apply(lambda x: x.split(','))

print('Simple weighted shortes paths:')
paths = graph.simple_weighted_shortest_path(test_cases=pyminer_test, data=data, method='mol_weight')
