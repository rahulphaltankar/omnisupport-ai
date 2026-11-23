# Tabular NN conclusion (generated)

- Best configuration: **B_adam_dropout_wd** (best epoch 20, validation PR-AUC 0.5751).
- Overfitting evidence: the unregularised large network E reaches a train loss of 0.4531 but validation loss 0.6375 (gap +0.1844; best epoch 4 of 25). The small unregularised net A has gap +0.0082; dropout + weight decay (B) gives -0.0016.
- Optimiser comparison (validation PR-AUC): Adam 0.5751, SGD+momentum 0.5713, AdamW+plateau 0.5750.
- Held-out test (threshold 0.23 from validation): NN PR-AUC 0.5727, ROC-AUC 0.7333 vs classical selected model PR-AUC 0.5720, ROC-AUC 0.7337.
- Reproducibility: re-running 3 epochs with the same seed changes validation loss by at most 0.00e+00.
