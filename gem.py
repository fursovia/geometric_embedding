"""
GEM algorithm
"""

import numpy as np
from typing import List
from embeddings import get_embedding_matrix, embedding_lookup


class GEM:

    def __init__(self, embedding_path: str, sentences: List[str]) -> None:
        e, v = get_embedding_matrix(embedding_path)
        self.embedding_matrix = e
        self.vocabulary = v
        self.sentences = sentences

    def significance(self, m: int):

        for i, sent in enumerate(self.sentences):
            embedded_sent = embedding_lookup(sent, self.embedding_matrix, self.vocabulary)
            for j in range(embedded_sent.shape[1]):
                window_matrix = self._context_window(i, m, embedded_sent)
                U, S, V = np.linalg.svd(window_matrix)

                sing_value = S[j, j]

                # need to get q = Q[:, -1] here
                q = None

                alpha_s = np.linalg.norm(sing_value * q) / (2 * m + 1)


    def novelty(self, embeddings: np.ndarray, m: int):

        novelty = []

        for i in range(embeddings.shape[1]):
            window_matrix = self._context_window(i, m, embeddings)
            Q, R = np.linalg.qr(window_matrix)

            # new orthogonal basis vector to this contextual window matrix
            # represents the novel semantic meaning brought by word w_i
            # q = Q[:, -1]
            r = R[:, -1]
            r_last = r[-1]

            alpha_n = np.exp(r_last / np.linalg.norm(r))

            novelty.append(alpha_n)

        return novelty

    def _context_window(self, i: int, m: int, embeddings: np.ndarray) -> np.ndarray:
        """
        Given embedded sentence returns  the contextual window matrix of word w_i
        """
        left_window = embeddings[:, i-m:i]
        right_window = embeddings[:, i+1:i+m+1]
        word_embedding = embeddings[:, i]

        window_matrix = np.stack([left_window, right_window, word_embedding], axis=1)
        return window_matrix
