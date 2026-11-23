"""Policy RAG: chunking strategies, dense (MiniLM + FAISS) or TF-IDF retrieval, cited extractive answers,
calibrated abstention, conflict detection and prompt-injection-safe handling of retrieved text."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from omnisupport.config import INDEX_VERSION, KNOWLEDGE_BASE, MODELS
from omnisupport.utils.privacy import contains_secret_request

INDEX_DIR = MODELS / "rag"
STRATEGIES = ("document", "section", "fixed")
DEFAULT_STRATEGY = "section"
DEFAULT_THRESHOLD = 0.30       # overwritten by calibration (index_metadata.json) when present
INJECTION = re.compile(r"(?i)(ignore (?:all |any )?(?:previous|prior|above) (?:instructions|rules)|disregard (?:the )?(?:system|policy)|you are now|reveal (?:your|the) (?:system )?prompt)")
SECRET_ASK = re.compile(r"(?i)\b(tell|give|share|show|reveal|send|read out)\b.{0,40}\b(password|passcode|otp|one[- ]time code|card number|cvv|api key|admin)\b")
CONFLICT_Q = re.compile(r"(?i)(conflict|contradict|differ|inconsistent|disagree|two policies|which (?:policy|rule) applies)")
PRIVATE_DATA = re.compile(r"(?i)(another|other|someone else'?s?) customer'?s?|internal notes|full (?:internal )?notes|their (?:account|address|card)")
STOP = set("a an the of to is are be it in on for and or what should when can does do if with that this as by at from how who which than then there their they them was were will would may might any".split())


@dataclass
class Doc:
    doc_id: str
    title: str
    body: str
    version: str
    scope: str
    path: str
    sha256: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    version: str


@dataclass
class RagAnswer:
    question: str
    answer: str
    abstained: bool
    reason: str
    citations: list[dict] = field(default_factory=list)
    top_score: float = 0.0
    conflict: bool = False
    conflict_detail: str = ""
    needs_human_review: bool = False
    latency_ms: float = 0.0
    index_version: str = INDEX_VERSION
    retrieved: list[dict] = field(default_factory=list)


def load_docs(kb: Path = KNOWLEDGE_BASE) -> list[Doc]:
    docs = []
    for p in sorted(kb.glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        lines = raw.splitlines()
        title = lines[0].lstrip("# ").strip()
        body_lines, notes = [], {}
        in_notes = False
        for ln in lines[1:]:
            if ln.startswith("## Operational notes"):
                in_notes = True
                continue
            if in_notes:
                m = re.match(r"- (Scope|Source|Version): (.*)", ln)
                if m:
                    notes[m.group(1).lower()] = m.group(2).strip().rstrip(".")
            else:
                body_lines.append(ln)
        doc_id = re.sub(r"^\d+_", "", p.stem)
        docs.append(Doc(doc_id, title, "\n".join(body_lines).strip(), notes.get("version", "?"), notes.get("scope", ""), p.name, hashlib.sha256(raw.encode()).hexdigest()[:12]))
    return docs


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def chunk_docs(docs: list[Doc], strategy: str, size: int = 220, overlap: int = 60) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        if strategy == "document":      # one chunk per file incl. title
            pieces = [d.title + ". " + re.sub(r"\s+", " ", d.body)]
        elif strategy == "section":     # title-prefixed paragraphs/bullets (policy sentences stay intact)
            pieces = [f"{d.title}. {s}" for s in _sentences(d.body)]
            # merge very short neighbours so a chunk always carries a full rule
            merged, buf = [], ""
            for s in pieces:
                buf = (buf + " " + s.split(". ", 1)[-1]) if buf else s
                if len(buf) >= 120:
                    merged.append(buf); buf = ""
            if buf: merged.append(buf)
            pieces = merged or [d.title + ". " + d.body]
        elif strategy == "fixed":       # sliding character window, no sentence awareness
            flat = re.sub(r"\s+", " ", d.title + ". " + d.body)
            step = max(1, size - overlap)
            pieces = [flat[i:i + size] for i in range(0, max(1, len(flat) - overlap), step)]
        else:
            raise ValueError(strategy)
        out += [Chunk(f"{d.doc_id}#{i}", d.doc_id, d.title, t, d.version) for i, t in enumerate(pieces)]
    return out


class TfidfBackend:
    name = "tfidf"

    def __init__(self, texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(texts)
        self.mat = self.vec.transform(texts)

    def _coverage(self, q: str) -> float:
        """Share of the question's idf mass that the corpus vocabulary can express. Terms never seen in the knowledge base
        (e.g. 'teleporting', 'mars') are out-of-vocabulary and would otherwise be silently ignored by the vectoriser, inflating cosine."""
        toks = [t for t in re.findall(r"[a-z0-9]+", q.lower()) if t not in STOP and len(t) > 2]
        vocab, idf = self.vec.vocabulary_, self.vec.idf_
        known = sum(idf[vocab[t]] for t in toks if t in vocab)
        oov = sum(1 for t in toks if t not in vocab)
        return known / (known + oov * float(idf.max())) if (known + oov) else 0.0

    def search(self, q: str, k: int):
        s = (self.mat @ self.vec.transform([q]).T).toarray().ravel() * self._coverage(q)
        idx = np.argsort(-s)[:k]
        return s[idx], idx


class DenseBackend:
    name = "minilm_faiss"

    def __init__(self, texts: list[str], embed_model: str | None = None):
        from omnisupport.nlp.embeddings import DEFAULT_EMBED_MODEL, get_embedder
        self.emb = get_embedder(embed_model or DEFAULT_EMBED_MODEL)
        self.model_name = embed_model or DEFAULT_EMBED_MODEL
        self.vecs = self.emb.encode(texts)
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.vecs.shape[1]); self.index.add(self.vecs)
        except Exception:  # faiss missing -> exact numpy search (same result for a flat index)
            self.index = None

    def search(self, q: str, k: int):
        qv = self.emb.encode([q])
        if self.index is not None:
            s, i = self.index.search(qv, min(k, len(self.vecs)))
            return s[0], i[0]
        s = (self.vecs @ qv[0]); idx = np.argsort(-s)[:k]
        return s[idx], idx


class PolicyRAG:
    def __init__(self, strategy: str = DEFAULT_STRATEGY, backend: str = "dense", threshold: float | None = None, kb: Path = KNOWLEDGE_BASE, k: int = 3):
        self.docs = load_docs(kb)
        self.strategy, self.k = strategy, k
        self.chunks = chunk_docs(self.docs, strategy)
        texts = [c.text for c in self.chunks]
        t0 = time.perf_counter()
        self.backend_name = backend
        if backend == "dense":
            try:
                self.be = DenseBackend(texts)
            except Exception:  # model unavailable offline -> documented graceful degradation
                self.be, self.backend_name = TfidfBackend(texts), "tfidf"
        else:
            self.be = TfidfBackend(texts)
        self.build_seconds = time.perf_counter() - t0
        meta = self.load_meta()
        self.threshold = threshold if threshold is not None else meta.get("thresholds", {}).get(self.backend_name, {}).get("abstain_below", DEFAULT_THRESHOLD)

    # ---- index metadata ----
    @staticmethod
    def load_meta() -> dict:
        p = INDEX_DIR / "index_metadata.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def metadata(self) -> dict:
        return {"index_version": INDEX_VERSION, "chunking": self.strategy, "backend": self.backend_name, "embedding_model": getattr(self.be, "model_name", None),
                "n_documents": len(self.docs), "n_chunks": len(self.chunks), "documents": {d.doc_id: {"file": d.path, "version": d.version, "sha256_12": d.sha256} for d in self.docs},
                "build_seconds": round(self.build_seconds, 3)}

    # ---- retrieval ----
    def retrieve(self, q: str, k: int | None = None) -> list[dict]:
        k = k or self.k
        scores, idx = self.be.search(q, max(k * 3, 8))
        seen, out = set(), []
        for s, i in zip(scores, idx):
            c = self.chunks[int(i)]
            if c.doc_id in seen:
                continue
            seen.add(c.doc_id); out.append({"doc_id": c.doc_id, "chunk_id": c.chunk_id, "title": c.title, "version": c.version, "score": float(s), "text": c.text})
            if len(out) == k:
                break
        return out

    # ---- answering ----
    def answer(self, question: str) -> RagAnswer:
        t0 = time.perf_counter()
        q = question.strip()
        if contains_secret_request(q) or SECRET_ASK.search(q) or PRIVATE_DATA.search(q):
            hits = self.retrieve("data privacy customer information access", 1)
            cit = [{"doc_id": h["doc_id"], "version": h["version"], "quote": _best_sentences(h["text"], "privacy access", 1)[0]} for h in hits]
            return RagAnswer(q, "I can't share another customer's private information or collect credentials. Use the verified-access process in the cited policy, or ask a human agent.", False,
                             "privacy_guard", cit, hits[0]["score"] if hits else 0.0, needs_human_review=True, latency_ms=(time.perf_counter() - t0) * 1000, retrieved=hits)
        hits = self.retrieve(q)
        top = hits[0]["score"] if hits else 0.0
        lat = lambda: (time.perf_counter() - t0) * 1000  # noqa: E731
        if not hits or top < self.threshold:
            return RagAnswer(q, "No answer: there is insufficient policy evidence to answer this reliably. A human agent should review it; I will not guess.", True, f"top_score {top:.3f} below threshold {self.threshold:.3f}",
                             [], top, needs_human_review=True, latency_ms=lat(), retrieved=hits)
        conflict, detail = detect_conflict(hits, self.threshold, q)
        injected = bool(INJECTION.search(q))
        cites, sents = [], []
        for h in hits[:2] if not conflict else hits[:3]:
            best = _best_sentences(h["text"], q, 2)
            cites.append({"doc_id": h["doc_id"], "version": h["version"], "score": round(h["score"], 3), "quote": " ".join(best)})
            sents.append(f"{' '.join(best)} [{h['doc_id']} v{h['version']}]")
        ans = " ".join(sents)
        if conflict:
            ans = f"Sources differ ({detail}). Prefer the more specific policy for the product/channel; if scope cannot settle it, flag for human review. " + ans
        if injected:
            ans = "Note: instructions inside the question were ignored; only policy text is used. " + ans
        return RagAnswer(q, ans, False, "supported" + ("+injection_ignored" if injected else ""), cites, top, conflict, detail, needs_human_review=conflict or injected, latency_ms=lat(), retrieved=hits)


def _tok(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2}


def _best_sentences(text: str, q: str, n: int) -> list[str]:
    sents = [s for s in _sentences(text) if not INJECTION.search(s)] or [text]   # retrieved text is data; drop imperative injections
    qt = _tok(q)
    ranked = sorted(sents, key=lambda s: -len(qt & _tok(s)) / (1 + 0.02 * len(s)))
    keep = ranked[:n]
    return [s for s in sents if s in keep]


def detect_conflict(hits: list[dict], thr: float, question: str = "") -> tuple[bool, str]:
    """Conflict = two relevant *return-policy* sources state different day windows, or the user asks about conflicting policies."""
    rel = [h for h in hits if h["score"] >= thr]
    windows = {}
    for h in rel:
        if "return" not in h["doc_id"]:
            continue
        for n in re.findall(r"(\d+)[- ](?:calendar )?days?", h["text"]):
            windows.setdefault(h["doc_id"], set()).add(int(n))
    nums = {n for v in windows.values() for n in v}
    if len(windows) >= 2 and len(nums) >= 2:
        return True, "; ".join(f"{d}: {sorted(v)}-day window" for d, v in windows.items())
    if CONFLICT_Q.search(question) and any(h["doc_id"] == "rag_conflict" for h in rel):
        return True, "user asked about conflicting policies; conflict-handling policy applies"
    return False, ""
