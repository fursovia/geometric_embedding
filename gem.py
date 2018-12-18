"""
GEM algorithm
"""

from typing import List
import numpy as np
from embeddings import inds_to_embeddings, sentence_to_indexes


def gram_schmidt_qr(A):
    nrows, ncols = A.shape
    Q = np.zeros((nrows, ncols))
    R = np.zeros((ncols, ncols))
    for j in range(ncols):
        u = np.copy(A[:, j])
        for i in range(j):
            proj = np.dot(A[:, j], Q[:, i]) * Q[:, i]
            u -= proj
        Q[:, j] = u / (np.linalg.norm(u, ord=2, axis=0) + 1e-18)

    for j in range(ncols):
        for i in range(j + 1):
            R[i, j] = A[:, j].dot(Q[:, i])

    return Q, R


class SentenceEmbedder:

    def __init__(self, sentences_raw: List[str], embedding_matrix: np.ndarray, vocab: dict, bigrams: bool = False):
        self.vocab = vocab
        self.sentences_raw = sentences_raw
        self.sentences = []
        self.bigrams = bigrams

        for sent in self.sentences_raw:
            self.sentences.append(sentence_to_indexes(sent, self.vocab))

        self.embedding_matrix = embedding_matrix
        self.emb_dim = self.embedding_matrix.shape[1]
        self.singular_values = None

    def gem(self, window_size: int = 7, k: int = 45, h: int = 17, sigma_power: int = 3):
        """
        Runs the GEM algorithm
        Returns:
            sentence_embeddings: shape [n, d]
        """
        X = np.zeros((self.emb_dim, len(self.sentences)))

        for i, sent in enumerate(self.sentences):
            embedded_sent = inds_to_embeddings(sent, self.embedding_matrix, self.bigrams)
            U, s, Vh = np.linalg.svd(embedded_sent, full_matrices=False)
            X[:, i] = U.dot(s ** sigma_power)

        D, s, _ = np.linalg.svd(X, full_matrices=False)
        self.singular_values = s.copy()
        D = D[:, :k]
        s = s[:k]

        C = np.zeros((self.emb_dim, len(self.sentences)))
        for j, sent in enumerate(self.sentences):
            embedded_sent = inds_to_embeddings(sent, self.embedding_matrix, self.bigrams)
            order = s * np.linalg.norm(embedded_sent.T.dot(D), axis=0)
            toph = order.argsort()[::-1][:h]
            alpha = np.zeros(embedded_sent.shape[1])
            for i in range(embedded_sent.shape[1]):
                window_matrix = self._context_window(i, window_size, embedded_sent)
                Q, R = gram_schmidt_qr(window_matrix)
                q = Q[:, -1]
                r = R[:, -1]
                alpha_n = np.exp(r[-1] / (np.linalg.norm(r, ord=2, axis=0)) + 1e-18)
                alpha_s = r[-1] / window_matrix.shape[1]  # (2 * window_size + 1)
                alpha_u = np.exp(-np.linalg.norm(s[toph] * (q.T.dot(D[:, toph]))) / h)
                alpha[i] = alpha_n + alpha_s + alpha_u
            C[:, j] = embedded_sent.dot(alpha)
            C[:, j] = C[:, j] - D.dot(D.T.dot(C[:, j]))

        sentence_embeddings = C.T
        return sentence_embeddings, self.singular_values

    def mean_embeddings(self):
        C = np.zeros((self.emb_dim, len(self.sentences)))

        for i, sent in enumerate(self.sentences):
            embedded_sent = inds_to_embeddings(sent, self.embedding_matrix, self.bigrams)
            C[:, i] = np.mean(embedded_sent, axis=1)

        return C.T, None

    def _context_window(self, i: int, m: int, embeddings: np.ndarray) -> np.ndarray:
        """
        Given embedded sentence returns  the contextual window matrix of word w_i
        """
        left_window = embeddings[:, i - m:i]
        right_window = embeddings[:, i + 1:i + m + 1]
        word_embedding = embeddings[:, i][:, None]
        window_matrix = np.hstack([left_window, right_window, word_embedding])
        return window_matrix
