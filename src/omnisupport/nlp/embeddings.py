"""Sentence embeddings with plain ``transformers`` (mean pooling + L2 norm), matching all-MiniLM-L6-v2 semantics."""
from __future__ import annotations

import numpy as np

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, max_length: int = 128, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_name, self.max_length = model_name, max_length
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.dim = int(self.model.config.hidden_size)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(list(texts[i:i + batch_size]), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                out.append(emb.cpu().numpy())
        return np.vstack(out).astype("float32") if out else np.zeros((0, self.dim), dtype="float32")


_CACHE: dict[str, SentenceEmbedder] = {}


def get_embedder(model_name: str = DEFAULT_EMBED_MODEL) -> SentenceEmbedder:
    if model_name not in _CACHE:
        _CACHE[model_name] = SentenceEmbedder(model_name)
    return _CACHE[model_name]
