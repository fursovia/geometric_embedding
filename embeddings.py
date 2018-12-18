"""
Things that help to work with embeddings
"""

from typing import List, Tuple

import numpy as np
from nltk import wordpunct_tokenize


class GloveEmbedder:
    def __init__(self, path_to_glove: str):
        self.matrix, self.vocab = get_embedding_matrix(path_to_glove)

    def get_vecs(self, tokens: List[str]) -> np.array:
        indices = tokens_to_indexes(tokens, self.vocab)
        vecs = np.array(inds_to_embeddings(indices, self.matrix))
        return vecs

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
            coefs = np.array(values[1:], dtype=np.float32)
            embeddings[word] = coefs
            vocabulary.append(word)

    embedding_matrix = np.array(list(embeddings.values()))
    vocabulary = {word: i for i, word in enumerate(vocabulary)}

    return embedding_matrix, vocabulary


def preprocess_sentence(sentence: str):
    """
    :return: list of words
    """
    return list(filter(str.isalpha, wordpunct_tokenize(sentence.lower())))


def tokens_to_indexes(words: List[str], vocab: dict) -> List[int]:
    indexes = []
    for word in words:
        if word in vocab:
            indexes.append(vocab[word])
    return indexes


def sentence_to_indexes(sentence: str, vocab: dict) -> List[int]:
    words = preprocess_sentence(sentence)
    indexes = []
    for word in words:
        if word in vocab:
            indexes.append(vocab[word])
    return indexes


def inds_to_embeddings(indexes: List[int], emb_matrix: np.ndarray) -> np.ndarray:
    # shape: [d, n] (embedding dim, number of words)
    return emb_matrix[indexes].T
