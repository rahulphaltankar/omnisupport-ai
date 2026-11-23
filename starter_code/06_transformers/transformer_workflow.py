"""Session 20-22 transformer starter.

Mandatory student work:
1. Demonstrate query/key/value, self-attention and positional information using
   one support message; include a worked example or visualisation and contrast
   it with TF-IDF.
2. Use pretrained transformer pipelines for at least two appropriate tasks.
3. Fine-tune/adapt a small Hugging Face text classifier on the supplied
   10,000-row subset and compare it with the classical TF-IDF baseline.
"""
from pathlib import Path
import pandas as pd

SUBSET_PATH = Path(__file__).resolve().parents[2] / "data" / "subsets" / "transformer_finetune_10000.parquet"


def load_finetuning_subset():
    return pd.read_parquet(SUBSET_PATH)


def prepare_training_subset(df, n=10000):
    """Deterministic helper only; model/tokenizer/training/evaluation remain student tasks."""
    return df.sample(min(n, len(df)), random_state=42).copy()


def demonstrate_attention(message: str):
    """TODO: create your Session 20 Q/K/V + self-attention/position evidence."""
    raise NotImplementedError("Student task: attention worked example or visualisation")


def fine_tune_text_classifier(train_df, validation_df):
    """TODO: tokenise, train/adapt a small HF classifier, evaluate and compare with TF-IDF."""
    raise NotImplementedError("Student task: mandatory Session 22 fine-tuning")
