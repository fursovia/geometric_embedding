"""
GEM algorithm
"""

from typing import List

import numpy as np

from embeddings import inds_to_embeddings


def gram_schmidt_qr(A):
    nrows, ncols = A.shape
    Q = np.zeros((nrows, ncols))
    R = np.zeros((ncols, ncols))
    for j in range(ncols):
        u = np.copy(A[:, j])
        for i in range(j):
            proj = np.dot(A[:, j], Q[:, i]) * Q[:, i]
            u -= proj
        Q[:, j] = u / np.linalg.norm(u)
    for j in range(ncols):
        for i in range(j + 1):
            R[i, j] = A[:, j].dot(Q[:, i])
    #     R = Q.T.dot(A)
    return Q, R

class GEM:

    def __init__(self, sentences: List[List[int]], embedding_matrix: np.ndarray) -> None:
        self.sentences = sentences  # List of idx representations of sentences
        self.embedding_matrix = embedding_matrix
        self.emb_dim = self.embedding_matrix.shape[1]

    def get_sentence_embeddings(self, window_size: int = 7, k: int = 45, h: int = 17, sigma_power: int = 3):
        X = np.zeros((self.emb_dim, len(self.sentences)))

        for i, sent in enumerate(self.sentences):
            embedded_sent = inds_to_embeddings(sent, self.embedding_matrix)
            U, s, Vh = np.linalg.svd(embedded_sent, full_matrices=False)
            X[:, i] = U.dot(s ** sigma_power)

        D, s, _ = np.linalg.svd(X, full_matrices=False)
        s_old = s.copy()
        D = D[:, :k]
        s = s[:k]

        C = np.zeros((self.emb_dim, len(self.sentences)))
        for j, sent in enumerate(self.sentences):
            embedded_sent = inds_to_embeddings(sent, self.embedding_matrix)
            order = s * np.linalg.norm(embedded_sent.T.dot(D), axis=0)
            toph = order.argsort()[::-1][:h]
            alpha = np.zeros(embedded_sent.shape[1])
            for i in range(embedded_sent.shape[1]):
                window_matrix = self._context_window(i, window_size, embedded_sent)
                Q, R = gram_schmidt_qr(window_matrix)
                q = Q[:, -1]
                r = R[:, -1]
                alpha_n = np.exp(r[-1] / np.linalg.norm(r))
                alpha_s = r[-1] / (2 * window_size + 1)
                alpha_u = np.exp(-np.linalg.norm(s[toph] * (q.T.dot(D[:, toph]))) / h)
                alpha[i] = alpha_n + alpha_s + alpha_u
            C[:, j] = embedded_sent.dot(alpha)
            C[:, j] = C[:, j] - D.dot(D.T.dot(C[:, j]))
        return C, s_old

    def _context_window(self, i: int, m: int, embeddings: np.ndarray) -> np.ndarray:
        """
        Given embedded sentence returns  the contextual window matrix of word w_i
        """
        left_window = embeddings[:, i - m:i]
        right_window = embeddings[:, i + 1:i + m + 1]
        word_embedding = embeddings[:, i][:, None]
        window_matrix = np.hstack([left_window, right_window, word_embedding])
        return window_matrix
