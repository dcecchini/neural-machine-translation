"""
Graph Networks for language modeling: Neural Machine Translation

@author: David Cecchini
@author2: Stece Beattie
"""

import re
from keras.layers.recurrent import LSTM
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Dense, RepeatVector, TimeDistributed, Bidirectional
import numpy as np
import pandas as pd
import pickle
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split
import nltk
import networkx as nx


WORDVEC_DIM = 100
EPOCHS = 200
BATCH_SIZE = 64
TEST_SIZE=0.2
INPUT_COLUMN = 0  # English is column 0, Portugues is column 1
TARGET_COLUMN = 1 # English is column 0, Portugues is column 1

MAX_VOCAB_SIZE = 5000 # Limit the vocabulary size for memory resons
OOV_TOKEN = r"<OOV>" # Out of vocabulary token

# Based on: https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html
import torch
import torch.nn as nn
import torch.nn.functional as F
class GATLayer():
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()

        self.batch_size = batch_size
        self.ngram = 2

        self.g = g
        # Linear transformation
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # Pair-wise un-normalized attention score between two neighbors
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)


    # GNN methods
    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF
        # Softmax
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # Aggregate neighbors
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h






class GNN():
    def __init__(self, batch_size, ngram=2):
        self.batch_size=batch_size
        self.ngram = ngram


    def _build_model(self):
        input_nx = nx.DiGraph(self.input_graph_table)
        target_nx = nx.DiGraph(self.target_graph_table)

        model = GAT(input_nx, in_dim=features.size()[1], hidden_dim=8, out_dim=7, num_heads=2)


    def _count_ngram(self, text_data):

        out_table = {}

        for bigrams in [nltk.ngrams(phrase.split(), n=self.ngram) for phrase in text_data]:
            for words in bigrams:
                if words[0] not in out_table.keys():
                    out_table[words[0]] = {}

                for i in range(1, len(words)):
                    if words[i] not in out_table[words[0]].keys():
                        out_table[words[0]][words[i]] = 1
                    else:
                        out_table[words[0]][words[i]] += 1

        return out_table


    def preprocess_text(self, text_data):
        """Creates a table for the graph network
        """

        input_texts = text_data[:, INPUT_COLUMN]
        target_texts = text_data[:, TARGET_COLUMN]

        input_dict = self._count_ngram(input_texts)
        target_dict = self._count_ngram(target_texts)

        input_table = []
        target_table = []

        for w, neigbors in input_dict.items():
            for n in neigbors:
                input_table.append((w, n))

        for w, neigbors in target_dict.items():
            for n in neigbors:
                target_table.append((w, n))

        self.input_graph_table = input_table
        self.target_graph_table = target_table

        return


    def save(self, filename):

        parameters = {  'input_graph_table': self.input_graph_table,
                        'target_graph_table': self.target_graph_table}

        with(open(filename, "wb")) as f:
            pickle.dump(parameters, f)

        return


    def load(self, filename):
        with(open(filename, "rb")) as f:
            parameters = pickle.load(f)

        self.input_graph_table = parameters['input_graph_table']
        self.target_graph_table = parameters['target_graph_table']
        return


    def train(self):
        pass


    def evaluate(self):
        pass


