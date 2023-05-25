'''
- First script to run. It preprocesses the raw data taken from pyminer paper and generates the datasets.
'''

import pandas as pd
from tqdm import tqdm
import json
from itertools import product
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

    # **************** Compounds data **********************
    df = pd.read_excel('data/original/KEGG_Pathway_Search_Ori.xlsx', sheet_name='Compound')

    # example usage
    element_names = extract_elements(df, 'Formula')
    print('The chemical elements that could be found in the given metabolites are:\n', element_names)

    # Create a col for every element
    for elm in element_names: df[elm]=0

    df['polymer'] = 0
    for row in range(len(df)):
        formula = df['Formula'].iloc[row]
        stoichiometry = extract_stoichiometry(formula)
        for key, value in stoichiometry.items():
            df.loc[df.index[row], key] = value
        if 'n' in df['Formula'].iloc[row]: 
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
        if 'n' in df['Formula'].iloc[row]: 
            df.loc[df.index[row], 'polymer'] = 1

    df.to_csv('data/compounds_final.csv')

    # **************** Reaction & Pairs data **********************
    df = pd.read_excel('data/original/KEGG_Pathway_Search_Ori.xlsx', sheet_name='Reaction')
    df.to_csv('data/reactions_final.csv')

    df['Compound']  = df['Compound'].apply(lambda x: json.loads(x))

    pairs = pd.DataFrame()
    reacs = []
    for i, data in enumerate(df['Compound']):
        combinations = list(product(data[0], data[1]))
        reaction = df['Entry'].iloc[i]
        # Create a list of dictionaries representing each row with the pairs
        rows = [{'Pairs': '_'.join(pair)} for pair in combinations]
        
        for j in range(len(combinations)):reacs.append(reaction)

        # Append each row to the pairs DataFrame
        pairs = pairs.append(rows, ignore_index=True)

    pairs['source'] = pairs['Pairs'].apply(lambda x: x.split('_')[0])
    pairs['target'] = pairs['Pairs'].apply(lambda x: x.split('_')[1])
    pairs['Reaction'] = reacs
    pairs.to_csv('data/pairs_final.csv')

if __name__ == '__main__':
    main()