"""
Things that help to work with embeddings
"""

from typing import Tuple

import numpy as np


def get_embedding_matrix(path_to_glove: str) -> Tuple[np.ndarray, dict]:
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


def embedding_lookup(sentence: str, emb_matrix: np.ndarray, vocab: dict) -> np.ndarray:
    """
    Embeds the sentence
    """
    words = sentence.strip().lower().split()
    indexes = []
    for word in words:
        if word in vocab:
            indexes.append(vocab[word])

    # shape: [d, n] (embedding dim, number of words)
    return emb_matrix[indexes].T
