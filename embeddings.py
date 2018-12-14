"""
Things that help to work with embeddings
"""

from typing import List, Tuple
from nltk import wordpunct_tokenize
import numpy as np


def get_glove_embedding_matrix(path_to_glove: str) -> Tuple[np.ndarray, dict]:
    """
    Function returns:
    1) Embedding matrix
    2) Vocabulary
    """

    embeddings = dict()
    vocabulary = []

    with open(path_to_glove, 'r') as file:
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
