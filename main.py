''' 
by comex
'''
# Import standard libraries
import os.path
from os import path

# Import Pandas, so we can use dataframes
import pandas as pd
import numpy as np
from tqdm import tqdm
import calculate_node_centralities

import matplotlib.pyplot as plt
import seaborn as sns

# load main datasets
pairs = pd.read_csv('data/pairs_final.csv', index_col=0)
df = pd.read_csv('data/compounds_final.csv', index_col=0)

source_mw = df.set_index('id')['mol_weight']
target_mw = source_mw.reindex(pairs['target']).values
pairs['MW'] = abs(source_mw.reindex(pairs['source']).values - target_mw) / (source_mw.reindex(pairs['source']).values + target_mw + 1e-6)
pairs['num_reactions'] = pairs['KEGG_reactions'].apply(lambda x: len(x.split(',')))
print('pairs:', pairs.shape, '\ncompounds:', df.shape)

for i in range(1, 11):
    print(f'Number of pairs existing in {i} reactios:', len(pairs[pairs['num_reactions'] == i]))
    
print()    
print('Number of pairs existing in more that 5 reactions:', len(pairs[pairs['num_reactions'] > 5]))

# *********** IMPORT TO NETWORKX *********************
import networkx as nx

def create_graph(pairs, df):
    G = nx.Graph()
    G = nx.from_pandas_edgelist(pairs, source='source', target='target')
    # Convert the df to dictionary
    node_data = df.set_index('id').to_dict('index')
    # Add node attributes to the graph
    nx.set_node_attributes(G, node_data)
    print('# nodes:', G.number_of_nodes(), "\n# edges:", G.number_of_edges())
    return G

def node_centralities(G, df):
    if path.exists('data/nodes_centralities.csv'):
        centralities_df = pd.read_csv('data/nodes_centralities.csv', index_col=0)
    else:
        centralities_df = calculate_node_centralities.calculate_centralities(G)    
        centralities_df.to_csv('data/nodes_centralities.csv')

    # ****************** Concat to df the centrality measures from dc ***********************
    df = pd.merge(df, centralities_df[['Node', 'PageRank', 'Degree Centrality']], left_on='id', right_on='Node')
    df.drop('Node', axis=1, inplace=True)
    print(df.shape)
    return df

# create graph
G = create_graph(pairs, df)
# calculate nodecentralities
df = node_centralities(G, df)

_temp = pairs.merge(df, left_on='source', right_on='id', suffixes=['_x', '_y'])
_temp.drop(['id', 'formula'], axis=1, inplace=True)
_temp = _temp.merge(df, left_on='target', right_on='id', suffixes=['_source', '_target'])
_temp.drop(['id', 'formula'], axis=1, inplace=True)
data = pd.DataFrame()
data['KEGG_reactions'] = _temp['KEGG_reactions']
data['num_reactions'] = _temp['num_reactions']
data['Reactant_pair'] = _temp['Reactant_pair'].copy()
data['source'] = _temp['source'].copy()
data['target'] = _temp['target'].copy()
data['MW'] = _temp['MW'].copy()

for col in ['W', 'I', 'Se', 'P', 'C', 'Cl', 'H', 'X',
       'R', 'Mo', 'N', 'As', 'Na', 'Co', 'B', 'Hg', 'Br', 'Ni', 'O', 'S', 'F',
       'Fe', 'Mn', 'Mg', 'polymer', 'mol_weight', 'Degree Centrality', 'PageRank']:
    data[col] = _temp[col+'_target'] - _temp[col+'_source']

del _temp

# Calculate total number of molecules echanged between target and source
data['total_molecules'] = data[['W', 'I', 'Se', 'P', 'C', 'Cl', 'H', 'X','R', 'Mo', 'N', 'As', 'Na', 'Co', 'B', 'Hg', 'Br', 'Ni', 'O', 'S', 'F','Fe', 'Mn', 'Mg']].abs().sum(axis=1)
print(data.head())
print(data.shape)
data.to_csv('data/curated_pairs.csv')

# read KEGG cofactors from PYMINER paper
cofactors = pd.read_csv('data/cofactors_KEGG.csv')
pyminer_cofactors = cofactors['Entry'].unique()
print('Number of cofactors: ', cofactors['Entry'].nunique())

pairs = data.copy()

# re-create graph
G = create_graph(pairs, df)

from tqdm import tqdm

def get_weights(a,b, method, centrality=False):
    
    # a_C and b_C is the number of C at each compound
    # Give huge weight if any of the compouds of a pair has no C molecules
    a_C = df[df['id']==a]['C'].values
    b_C = df[df['id']==b]['C'].values
    if (a_C != 0 and b_C == 0) or (a_C == 0 and b_C != 0):
        return 999
    
    # give huge weight if we have a cofactor
    if (a in pyminer_cofactors) or (b in pyminer_cofactors):
        return 999
        
    try:
        w = pairs[(pairs['source'] == a) & (pairs['target'] == b)][method].values[0]
    except IndexError:
        w = pairs[(pairs['source'] == b) & (pairs['target'] == a)][method].values[0]

    if(centrality):
        centr_w = (df[df['id']==a]['Degree Centrality'].values[0] + df[df['id']==b]['Degree Centrality'].values[0]) / 2
        w = (w + centr_w) * (1 - abs(w - centr_w))/2
                
    return w

# weight the graph edges based on MW and node centralities
for edge in tqdm(G.edges()):
    G.edges[(edge[0], edge[1])]['weight'] = get_weights(edge[0], edge[1], method='MW', centrality=True)
    
def get_reactions(a,b):
    try:
        w = pairs[(pairs['source'] == a) & (pairs['target'] == b)]['KEGG_reactions'].values[0]
    except IndexError:
        w = pairs[(pairs['source'] == b) & (pairs['target'] == a)]['KEGG_reactions'].values[0]
    return w
    
def get_number_reactions(a,b):
    try:
        w = pairs[(pairs['source'] == a) & (pairs['target'] == b)]['num_reactions'].values[0]
    except IndexError:
        w = pairs[(pairs['source'] == b) & (pairs['target'] == a)]['num_reactions'].values[0]
    return w
    
for edge in tqdm(G.edges()):
    G.edges[(edge[0], edge[1])]['reactions'] = get_reactions(edge[0], edge[1])
    G.edges[(edge[0], edge[1])]['num_reactions'] = get_reactions(edge[0], edge[1])




# ************** TEST THE PATWAY FINDER ON VALIDATION SETS ***********************
def simple_weighted_shortest_path(test_cases):
    correct_pathways = []
    for row in range(len(test_cases)):
        source = test_cases['source'].iloc[row]
        target = test_cases['target'].iloc[row]
        correct_pathways.append((list(nx.shortest_path(G, source, target, weight='weight')) == test_cases['paths_list'].iloc[row]))

    print(f'Correct pathway predictions: {correct_pathways.count(True)}')
    print(f'Correct pathway predictions (%): {100 * correct_pathways.count(True) / len(correct_pathways)}')
    return correct_pathways

def constrained_shortest_path(test_cases):
    BEST = []
    for row in range(len(test_cases)):
        source = test_cases['source'].iloc[row]
        target = test_cases['target'].iloc[row]

        paths = nx.shortest_simple_paths(G, source, target, weight='weight')
        
        BEST_SHORTEST = []
        for idx, val in enumerate(paths):
            if idx == 10: break
                
            g = G.subgraph(val)

            RXN = []
            for edge in g.edges(data=True):
                RXN.append(edge[2]['reactions'])
                    
            if (len(RXN) != len(set(RXN))) & ('nan' not in RXN): 
                pass
            else: 
                BEST_SHORTEST.append(val)
        
        if BEST_SHORTEST == []:
            for i, v in enumerate(paths):
                BEST_SHORTEST = v
                if i == 0: break

        BEST.append(BEST_SHORTEST[0])
        
    CORRECT = []
    for i in range(len(test_cases)):
        CORRECT.append(BEST[i] == test_cases['Pathway '].iloc[i].split(','))
        
    print(f'Correct pathway predictions: {CORRECT.count(True)}')
    print(f'Correct pathway predictions (%): {100 * CORRECT.count(True) / len(CORRECT)}')

    return BEST


''' Load file with test cases from NICEpath '''
test_cases = pd.read_csv('data/test_cases.csv')
test_cases['source'] = test_cases['Pathway '].apply(lambda x: x.split(',')[0])
test_cases['target'] = test_cases['Pathway '].apply(lambda x: x.split(',')[len(x.split(','))-1])
test_cases['paths_list'] = test_cases['Pathway '].apply(lambda x: x.split(','))

print('Simple weighted shortes paths:')
correct_pathways = simple_weighted_shortest_path(test_cases)
print('Constrained shortest paths:')
constrained_paths = constrained_shortest_path(test_cases=test_cases)

######### NEW VALIDATION SET ###########
pyminer_test = pd.read_csv('data/pyminer_validation_set.csv', delimiter=';', header=None, names=['Pathway'])
pyminer_test['source'] = pyminer_test['Pathway'].apply(lambda x: x.split(',')[0])
pyminer_test['target'] = pyminer_test['Pathway'].apply(lambda x: x.split(',')[len(x.split(','))-1])

print('Simple weighted shortes paths:')
correct_pathways = simple_weighted_shortest_path(pyminer_test)
print('Constrained shortest paths:')
constrained_paths = constrained_shortest_path(pyminer_test)
