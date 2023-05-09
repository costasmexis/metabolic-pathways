'''
By comex
'''
import pandas as pd
from tqdm import tqdm
import json
import re

def extract_elements(df, column_name):
    '''
    Function to extract the chemical elements that exist in the compounds
    '''
    # define the regular expression pattern to match the chemical formula
    pattern = r'[A-Z][a-z]?'
    # initialize a set to store the element symbols
    elements = set()
    # loop over the values in the specified column of the DataFrame
    for value in df[column_name].values:
        # find all matches of the pattern in the value string
        matches = re.findall(pattern, value)
        # add the matches to the set of elements
        elements.update(matches)
    return elements

def extract_stoichiometry(formula):
    '''
    Exctracts the stoichiometry 
    '''
    # define the regular expression pattern to match the chemical formula
    pattern = r'([A-Z][a-z]?)(\d*)'
    # initialize the dictionary to store the element symbol and its stoichiometry
    stoichiometry = {}
    # loop over the matches of the pattern in the formula string
    for match in re.findall(pattern, formula):
        symbol, count = match
        # if the count is empty, set it to 1
        count = int(count) if count else 1
        # add the symbol and count to the stoichiometry dictionary
        stoichiometry[symbol] = count
    return stoichiometry

def main():
    ''' Create compounds dataset '''
    _df = pd.read_excel('data/original/KEGG_Pathway_Search_Ori.xlsx', sheet_name='Compound')
    df = _df[['Entry','Formula','Mol Weight']].copy()
    df.rename({'Entry':'id', 'Formula':'formula', 'Mol Weight':'mol_weight'}, axis=1, inplace=True)

    # example usage
    element_names = extract_elements(df, 'formula')
    print('The chemical elements that could be found in the given metabolites are:\n', element_names)

    # Create a col for every element
    for elm in element_names: df[elm]=0

    for row in range(len(df)):
        formula = df['formula'].iloc[row]
        stoichiometry = extract_stoichiometry(formula)
        for key, value in stoichiometry.items():
            #df[key].iloc[row] = value
            df.loc[df.index[row], key] = value

    # Col that contains the info if the compound is a polymer or not
    df['polymer'] = 0

    for row in range(len(df)):
        if 'n' in df['formula'].iloc[row]: 
            df.loc[df.index[row], 'polymer'] = 1
            
    # dict of chemical elements and their molecular weight
    elements = {
        'Co': 58.93,
        'Se': 78.96,
        'Cl': 35.45,
        'Ni': 58.69,
        'N': 14.01,
        'Hg': 200.6,
        'B': 10.81,
        'F': 19.00,
        'Fe': 55.85,
        'Br': 79.90,
        'W': 183.8,
        'Mo': 95.94,
        'Mn': 54.94,
        'I': 126.9,
        'C': 12.01,
        'Na': 22.99,
        'H': 1.008,
        'O': 16.00,
        'S': 32.07,
        'As': 74.92,
        'P': 30.97,
        'Mg': 24.31
    }

    # calculate the molecular weights of every compound
    mw = []
    for row in tqdm(range(len(df))):
        weight = 0
        for col in elements.keys():
            weight = weight + elements[col] * df.iloc[row][col]
        if (df.iloc[row]['R'] + df.iloc[row]['polymer']) != 0:
            mw.append(weight * (df.iloc[row]['R'] + df.iloc[row]['polymer'])/2)
        else:
            mw.append(weight)
            
    df['mol_weight'] = mw

    # Col that contains the info if the compound is a polymer or not
    df['polymer'] = 0
    for row in range(len(df)):
        if 'n' in df['formula'].iloc[row]: 
            df.loc[df.index[row], 'polymer'] = 1

    df.to_csv('data/compounds_final.csv')


    ''' *********** Create pairs dataset ************** '''
    # Load the Excel file into a pandas DataFrame
    rxns = pd.read_excel('data/original/KEGG_Pathway_Search_Ori.xlsx', sheet_name='Reaction')
    # drop unusefull columns
    rxns.drop(columns=['Names', 'Definition', 'Direction', 'Coefficient',
        'SMILES',  'Status', 'Comment', 'EC Number', 'Rhea', 'Reference', 'Compound Pair (0.1)', 'Compound Pair (0.2)',
        'Compound Pair (0.3)', 'Compound Pair (0.4)', 'Compound Pair (0.5)', 'Compound Pair (0.6)', 'Compound Pair (0.7)', 'Compound Pair (0.8)',
        'Compound Pair (0.9)',], inplace=True)
            
    rxns.rename({'Compound Pair (1.0)':('Reaction_pair')}, axis=1, inplace=True)
    rxns['Reaction_pair'] = rxns['Reaction_pair'].apply(lambda x: json.loads(x))

    pairs = []
    reactions = []
    for i, row in enumerate(range(len(rxns))):
        for d in (rxns['Reaction_pair'][row]):
            for key, values in d.items():
                for value in values:
                    pairs.append(f"{key}_{value}")
                    reactions.append(rxns.iloc[i]['Entry'])

    pairs = pd.DataFrame(pairs, columns=['Reactant_pair'])
    pairs['KEGG_reactions'] = reactions
    pairs['source'] = pairs['Reactant_pair'].apply(lambda x: x.split('_')[0])
    pairs['target'] = pairs['Reactant_pair'].apply(lambda x: x.split('_')[1])
    # group by Reactant_pair and combine KEGG_reactions values with comma-separated values
    grouped = pairs.groupby('Reactant_pair')['KEGG_reactions'].apply(lambda x: ','.join(x)).reset_index()
    # select the first row of each group as the row to include in the final dataframe
    pairs = pairs.groupby('Reactant_pair').first().reset_index()
    # add the combined KEGG_reactions column to the final dataframe
    pairs['KEGG_reactions'] = grouped['KEGG_reactions']
    # sort the final dataframe by Reactant_pair and reset the index
    pairs = pairs.sort_values('Reactant_pair').reset_index(drop=True)
    pairs.reset_index(drop=True, inplace=True)
    pairs['CAR'] = -999
    pairs['RPAIR_main'] = True

    idx_drop = pairs[pairs['source'] == pairs['target']].index # Drop pairs where source==target
    pairs.drop(idx_drop, inplace=True)
    print(pairs.shape)

    kegg_pairs = pd.read_csv('data/original/kegg_pairs.csv', sep='\t')
    kegg_pairs['source'] = kegg_pairs['Reactant_pair'].apply(lambda x: x.split('_')[0])
    kegg_pairs['target'] = kegg_pairs['Reactant_pair'].apply(lambda x: x.split('_')[1])
    
    # concatenate the dataframes vertically
    concatenated = pd.concat([pairs, kegg_pairs], ignore_index=True)

    source_mw = df.set_index('id')['mol_weight']
    target_mw = source_mw.reindex(concatenated['target']).values

    concatenated['MW'] = abs(source_mw.reindex(concatenated['source']).values - target_mw) / (source_mw.reindex(concatenated['source']).values + target_mw + 1e-6)

    concatenated['num_reactions'] = concatenated['KEGG_reactions'].apply(lambda x: len(x.split(',')))
    
    print(concatenated.shape)
    concatenated.to_csv('data/pairs_final.csv')

if __name__ == '__main__':
    main()
