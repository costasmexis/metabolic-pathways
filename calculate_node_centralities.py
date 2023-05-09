import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def calculate_centralities(G, scaling=False):
    '''
    Function to calculate centralities for nodes of a given graph and
    store them to a single dataframe
    '''
    pr = nx.pagerank(G)
    pr = pd.DataFrame(list(pr.items()), columns=['Node', 'PageRank'])

    dc = nx.degree_centrality(G)
    dc = pd.DataFrame(list(dc.items()), columns=['Node', 'Degree Centrality'])

    # bc = nx.betweenness_centrality(G)
    # bc = pd.DataFrame(list(bc.items()), columns=['Node', 'Betweenness Centrality'])

    # centralities
    centralities_df = dc.copy()
    centralities_df['PageRank'] = pr['PageRank'].copy()
    # centralities_df['Betweenness Centrality'] = bc['Betweenness Centrality'].copy()

    centralities_df.sort_values(by='PageRank', ascending=False, inplace=True)

    if scaling:
        scaler = MinMaxScaler()
        # centralities_df[['Degree Centrality', 'PageRank', 'Betweenness Centrality']] = scaler.fit_transform(centralities_df[['Degree Centrality', 'PageRank', 'Betweenness Centrality']])
        centralities_df[['Degree Centrality', 'PageRank']] = scaler.fit_transform(centralities_df[['Degree Centrality', 'PageRank']])

    return centralities_df