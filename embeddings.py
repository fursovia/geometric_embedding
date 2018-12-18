"""
Things that help to work with embeddings
"""

from typing import List, Tuple
import numpy as np
from utils import preprocess_sentence


class GloveEmbedder:
    def __init__(self, path_to_glove: str):
        self.matrix, self.vocab = get_embedding_matrix(path_to_glove)

    def get_vecs(self, tokens: List[str]) -> np.array:
        indices = tokens_to_indexes(tokens, self.vocab)
        return inds_to_embeddings(indices, self.matrix)

    def get_vecs_average(self, tokens: List[str]) -> np.array:
        return np.mean(self.get_vecs(tokens), axis=1)

    def __call__(self, tokens):
        return self.get_vecs_average(tokens)


def get_embedding_matrix(path: str, islexvec: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Function returns:
    1) Embedding matrix
    2) Vocabulary
    """

    embeddings = dict()
    vocabulary = []

    with open(path, 'r') as file:
        if islexvec:
            file.readline()
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.array(values[1:], dtype=np.float64)
            embeddings[word] = coefs
            vocabulary.append(word)

    embedding_size = list(embeddings.values())[1].shape[0]

    embedding_matrix = np.zeros((len(vocabulary) + 1, embedding_size))
    embedding_matrix[-1] = np.mean(np.array(list(embeddings.values())), axis=0)

    vocab = dict()
    vocab['UNKNOWN_TOKEN'] = len(vocabulary)
    for i, word in enumerate(vocabulary):
        embedding_matrix[i] = embeddings[word]
        vocab[word] = i

    return embedding_matrix, vocab


def tokens_to_indexes(words: List[str], vocab: dict) -> List[int]:
    indexes = []
    for word in words:
        if word in vocab:
            indexes.append(vocab[word])
    return indexes


def sentence_to_indexes(sentence: str, vocab: dict) -> List[int]:
    tokens = preprocess_sentence(sentence)
    indexes = []
    for token in tokens:

        if token in vocab:
            indexes.append(vocab[token])
        else:
            indexes.append(vocab['UNKNOWN_TOKEN'])

    return indexes


def inds_to_embeddings(indexes: List[int], emb_matrix: np.ndarray, bigrams: bool = False) -> np.ndarray:
    if bigrams:
        embedded_sent = emb_matrix[indexes]
        if len(indexes) % 2 == 0:
            embedded_sent = (embedded_sent[::2] + embedded_sent[1::2]) / 2
        else:
            first = embedded_sent[::2]
            second = embedded_sent[1::2]
            padded = np.zeros(first.shape)
            padded[:-1] = second

            embedded_sent = (first + padded) / 2
        return embedded_sent.T
    # shape: [d, n] (embedding dim, number of words)
    return emb_matrix[indexes].T
