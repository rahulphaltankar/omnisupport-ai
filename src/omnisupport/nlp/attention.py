"""Attention building blocks used by the Session-20 demonstration (pure NumPy + real DistilBERT weights)."""
from __future__ import annotations

import numpy as np


def sinusoidal_positions(n_pos: int, d: int) -> np.ndarray:
    """Original Transformer positional encoding: PE[pos,2i]=sin(pos/10000^(2i/d)), PE[pos,2i+1]=cos(...)."""
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d)
    pe = np.zeros((n_pos, d))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def self_attention(X: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, bq=0.0, bk=0.0, bv=0.0):
    """Single-head scaled dot-product self-attention. Returns dict with Q, K, V, raw scores, weights and output."""
    Q, K, V = X @ Wq + bq, X @ Wk + bk, X @ Wv + bv
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    A = softmax(scores, axis=-1)
    return {"Q": Q, "K": K, "V": V, "raw_scores": Q @ K.T, "scaled_scores": scores, "weights": A, "output": A @ V, "d_k": d_k}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
