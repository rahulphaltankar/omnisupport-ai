"""RAG entry points: adapter over src/omnisupport/rag/pipeline.py (section chunking, MiniLM+FAISS, cited extractive answers, calibrated abstention)."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from omnisupport.config import KNOWLEDGE_BASE  # noqa: E402
from omnisupport.rag.pipeline import PolicyRAG  # noqa: E402


def build_index(policy_dir=KNOWLEDGE_BASE):
    return PolicyRAG(kb=policy_dir)


def answer_question(question, index):
    return index.answer(question)
