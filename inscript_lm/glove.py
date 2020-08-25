import torch
import numpy as np
from numpy import asarray
import itertools

"""
To download the GloVe embeddings, do the following:
wget nlp.stanford.edu/data/glove.6B.zip 
unzip glove.6B.zip

I use only the 50-dimensional ones, so glove.6B.50d.txt.
"""

def load_glove_embeddings(vocab):
    """
    Params:
    vocab : dict : vocabulary, token as key, index as value
    Load GloVe embedding vectors for all words in our vocabulary.
    Returns a dictionary word:vector
    #https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    """
    embeddings_index = dict()
    with open('/home/CE/skrjanec/glove.6B.50d.txt', 'r', encoding="utf-8") as f: # TODO fix path to file
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                coefs = asarray(values[1:], dtype='float32')
                embeddings_index[vocab[word]] = coefs

    return embeddings_index

def get_glove_embeddings(glove_embeddings, ntoken, embsize):
    """
    Params:
    glove embeddings : dict : word index to GloVe embedding vector
    ntoken: int : number of words in vocabulary
    embsize : int : embedding dimension

    Create embedding matrix used in nnmodel.py
    If a word was not in GloVe, initialize it randomly
    Returns a torch tensor with dims ntoken x embedding dimension
    """
    emb_matrix = np.empty(shape=(ntoken, embsize))
    for idx in range(ntoken):
        if idx in glove_embeddings:
            emb_matrix[idx] = glove_embeddings[idx]
        else:
            emb_matrix[idx] = np.random.normal(size=embsize)

    return torch.FloatTensor(emb_matrix)