"""
Things that help to work with embeddings
"""

import numpy as np
from typing import Tuple


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
        try:
            indexes.append(vocab[word])
        except KeyError:
            pass

    embeddings = []

    for idx in indexes:
        embeddings.append(emb_matrix[idx])

    embeddings = np.array(embeddings, dtype=np.float32).reshape(-1, emb_matrix.shape[1])

    return embeddings


if __name__ == '__main__':
    EMB_PATH = 'data/glove.6B.50d.txt'
    TEST_SENT = 'hello i love having a good time'

    e, v = get_embedding_matrix(EMB_PATH)
    print(len(v))
    print(e.shape)

    embedded_sentence = embedding_lookup(TEST_SENT, e, v)

    print(embedded_sentence)
    print(embedded_sentence.shape)

