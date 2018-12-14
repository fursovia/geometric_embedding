"""
GEM algorithm
"""

from typing import List

import numpy as np

from embeddings import inds_to_embedding


class GEM:

    def __init__(self, sentences: List[List[int]], embedding_matrix: np.ndarray) -> None:
        self.sentences = sentences  # List of idx representations of sentences
        self.embedding_matrix = embedding_matrix
        self.dim = self.embedding_matrix.shape[1]

    def get_sentence_embeddings(self, m: int, k: int, h: int):
        X = np.zeros((self.dim, len(self.sentences)))

        for i, sent in enumerate(self.sentences):
            embedded_sent = inds_to_embedding(sent, self.embedding_matrix)
            U, s, Vh = np.linalg.svd(embedded_sent, full_matrices=False)
            X[:, i] = U.dot(s ** 3)

        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        D = U[:, :k].dot(np.diag(s[:k]))

        C = np.zeros((self.embedding_matrix.shape[1], len(self.sentences)))
        for j, sent in enumerate(self.sentences):
            embedded_sent = inds_to_embedding(sent, self.embedding_matrix)
            order = s * np.linalg.norm(embedded_sent.T.dot(D), axis=0)
            toph = order.argsort()[::-1][:h]
            alpha = np.zeros(embedded_sent.shape[1])
            for i in range(embedded_sent.shape[1]):
                window_matrix = self._context_window(i, m, embedded_sent)
                Q, R = np.linalg.qr(window_matrix)
                q = Q[:, -1]
                r = R[:, -1]
                alpha_n = np.exp(r[-1] / np.linalg.norm(r))
                alpha_s = r[-1] / (2 * m + 1)
                alpha_u = np.exp(-np.linalg.norm(s[toph] * (q.T.dot(D[:, toph]))) / h)
                alpha[i] = alpha_n + alpha_s + alpha_u
            C[:, j] = embedded_sent.dot(alpha)
            C[:, j] = C[:, j] - D.dot(D.T.dot(C[:, j]))
        return C

    def _context_window(self, i: int, m: int, embeddings: np.ndarray) -> np.ndarray:
        """
        Given embedded sentence returns  the contextual window matrix of word w_i
        """
        left_window = embeddings[:, i - m:i]
        right_window = embeddings[:, i + 1:i + m + 1]
        word_embedding = embeddings[:, i][:, None]
        window_matrix = np.hstack([left_window, right_window, word_embedding])
        return window_matrix
