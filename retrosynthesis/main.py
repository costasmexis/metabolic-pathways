import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import torch
from torch import nn
import torch.nn.functional as F
import random

import sys
sys.path.append('/home/WUR/g0012069/metabolic-pathways/retrosynthesis/src')

from smiles_lstm.model.smiles_vocabulary import SMILESTokenizer, Vocabulary, create_vocabulary

from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
# ignore some deprecation warnings
warnings.filterwarnings('ignore')


def pad_sequence(tokenizer_array, desired_length):
    padded_sequence = pad_sequences([tokenizer_array], maxlen=desired_length, padding='post')[0]
    return padded_sequence

def preprocess_smiles_data(x):
    x = tk.tokenize(x, with_begin_and_end=False)
    x = vocabulary.encode(x )
    x  = pad_sequence(x, 160 )
    x  = torch.tensor([x ])
    
    return X

def canonicalize_smiles(smiles):
    '''This function takes a non-canonical SMILES and
    returns the canonical version'''
    mol = Chem.MolFromSmiles(smiles) #create a mol object from input smiles 
    canonical_smiles = Chem.MolToSmiles(mol) #convert the previous mol object to SMILES using Chem.MolToSmiles()
    return canonical_smiles

df = pd.read_csv('retrosynthesis/retrosynthesis-all', header=None)
df['source'] = df[0].apply(lambda x: x.split('>>')[0])
df['target'] = df[0].apply(lambda x: x.split('>>')[1])
df['source'] = df['source'].apply(lambda x: canonicalize_smiles(x))
df['target'] = df['target'].apply(lambda x: canonicalize_smiles(x))
df.drop(0, axis=1, inplace=True)
# Remove spaces from all columns
for col in df.columns: df[col] = df[col].str.replace(' ', '')

class Dataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset that takes a file containing \n separated SMILES.
    """

    def __init__(self, smiles_list : list, vocabulary : Vocabulary,
                 tokenizer : SMILESTokenizer) -> None:
        self._vocabulary  = vocabulary
        self._tokenizer   = tokenizer
        self._smiles_list = list(smiles_list)

    def __getitem__(self, i : int) -> torch.Tensor:
        smi     = self._smiles_list[i]
        tokens  = self._tokenizer.tokenize(smi, with_begin_and_end=False)
        encoded = self._vocabulary.encode(tokens)
        encoded = pad_sequence(encoded, 200)
        return torch.tensor(encoded.astype(int), dtype=torch.long)  # pylint: disable=E1102

    def __len__(self) -> int:
        return len(self._smiles_list)

    @staticmethod
    def collate_fn(encoded_seqs : list) -> torch.Tensor:
        """
        Converts a list of encoded sequences into a padded tensor.
        """
        max_length   = max([seq.size(0) for seq in encoded_seqs])
        collated_arr = torch.zeros(len(encoded_seqs),
                                   max_length,
                                   dtype=torch.long)  # padded with zeros
        for i, seq in enumerate(encoded_seqs):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr

# create a vocabulary using all SMILES in df
dataset = df['source'].unique().tolist() + df['target'].unique().tolist()
dataset = np.unique(dataset).tolist()

tokenizer = SMILESTokenizer()
vocabulary   = create_vocabulary(smiles_list=dataset, tokenizer=tokenizer, canonical=True)

print(f'There are {len(vocabulary)} unique tokens in the vocabulary.\n')

from sklearn.model_selection import train_test_split

print('Original dataset:')
print(df.shape)

# Splitting the data into train and combined val/test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Printing the sizes of the resulting splits
print("Train data size:", len(train_data))
print("Test data size:", len(test_data))

train     = train_data.copy()
test      = test_data.copy()

# train = train[:2000]
# test = test[:500]

train_X = Dataset(smiles_list=train['source'].tolist(), vocabulary=vocabulary, tokenizer=tokenizer)
train_X = train_X.collate_fn(train_X)
train_y = Dataset(smiles_list=train['target'].tolist(), vocabulary=vocabulary, tokenizer=tokenizer)
train_y = train_y.collate_fn(train_y)

test_X = Dataset(smiles_list=test['source'].tolist(), vocabulary=vocabulary, tokenizer=tokenizer)
test_X = test_X.collate_fn(test_X)
test_y = Dataset(smiles_list=test['target'].tolist(), vocabulary=vocabulary, tokenizer=tokenizer)
test_y = test_y.collate_fn(test_y)

print(train_X.shape, train_y.shape)
print(test_X.shape, test_y.shape)

import torch
import torch.nn as nn
import torch.optim as optim
import random
from torchtext.data import Field, BucketIterator

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, batch_size)
        embedding = self.dropout(self.embedding(x)) 
        # embedding shape: (seq_length, batch_size, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # shape of x: (batch_size) but we want (1, batch_size)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, bacth_size, embedding_size)

        outputs, (hidden, cell) = self. rnn(embedding, (hidden, cell))
        # shape of outputs: (1, batch_size, hidden_size)

        predictions = self.fc(outputs)
        # shape of predictions: (1, batch_size, length_of_vocab)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(vocabulary)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # Grab start token
        x = target[0]
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output
            
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
    
#### Training loop

# Training Hyperparameters
num_epochs = 3
learning_rate = 0.001
batch_size = 2048

# Model hyperparameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(vocabulary)
input_size_decoder = len(vocabulary)
output_size = len(vocabulary)
encoder_embedding_size = 5
decoder_embedding_size = 5
hidden_size = 10
num_layers = 1
enc_dropout = 0.5
dec_dropout = 0.5

encoder_net = Encoder(input_size_encoder, encoder_embedding_size,
                      hidden_size, num_layers, enc_dropout).to(device)

decoder_net = Decoder(input_size_decoder, decoder_embedding_size,
                      hidden_size, output_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0
    model.train()
    for i in tqdm(range(0, len(train_X), batch_size)):
        # Prepare batch
        batch_X = train_X[i:i+batch_size].to(device)
        batch_y = train_y[i:i+batch_size].to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(batch_X, batch_y)
        # output shape: (trg_len, batch_size, output_dim)
        print(output.shape)

