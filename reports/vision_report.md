# Return-image classification: basic CNN vs transfer learning

Script `scripts/07_vision.py` · Evidence `artifacts/vision/` · Weights `models/vision/` (`basic_cnn.pt` committed-size; ResNet weights git-ignored).

Data: 480 synthetic images, 4 balanced classes (damaged / normal / packaging_damage / wrong_item), stratified **70/15/15 = 336 / 72 / 72**, identical split for both models, seeds 42/43/44.
Models: custom 4-block CNN (147k parameters) trained from scratch with augmentation (affine, flip, brightness/contrast, noise); ResNet18 pre-trained on ImageNet with only `layer4` + new head trainable (8.4M of 11.2M parameters), inputs resized to 128.

| model | test accuracy (3 seeds) | macro-F1 | train s (mean) | ms / image |
|---|---:|---:|---:|---:|
| basic CNN | 1.000 ± 0 | 1.000 | 155 | 3.1 |
| ResNet18 transfer | 1.000 ± 0 | 1.000 | 88 | 10.8 |

**Both are perfect on the clean 72-image test set, so the clean test cannot rank them** (a 72-image set resolves differences of only ~1.4 points per image, and the images are synthetic and templated). We therefore added evidence that *can* separate them:

**Data-size curve** (`data_size_curve.csv`): with 10 % of the training data (32 images) the CNN collapses to macro-F1 0.52 while ResNet18 stays at 0.99; from 25 % upward both are ≥ 0.99. Pre-training matters when labels are scarce.

**Robustness to corruptions never seen in training** (same trained models, test split; `robustness.csv`, `fig_robustness.png`):

| corruption | CNN macro-F1 | ResNet18 macro-F1 |
|---|---:|---:|
| none / noise 0.05 / occlusion 35 % | 1.00 | 1.00 / 1.00 / 0.99 |
| blur σ=1 / σ=2 | 0.77 / 0.81 | 1.00 / 0.99 |
| rotate 20° / 40° | 0.57 / 0.10 | 0.99 / 0.91 |
| darken ×0.5 | 0.67 | 0.91 |
| **noise 0.15** | **1.00** | **0.77** |

Transfer learning is far more robust to blur, rotation and lighting; the CNN is better under heavy pixel noise (ResNet18 then predicts `damaged` for normal/packaging images: 8 + 8 errors, `error_confusion_pairs.csv`). Under 40° rotation the CNN predicts `packaging_damage` for every image of three classes — it learned a rotation-specific cue.

**Error analysis** is therefore about *conditions*, not clean errors (there are none): `fig_error_gallery.png` shows failing examples per condition. Practical implication: the served model is the CNN (3 ms/image, perfect on in-distribution images) with `low_confidence` routing to a human; if real photos vary in angle or lighting, switch to the ResNet18 weights.

**Limits.** Synthetic, small, class-balanced images; clean-test accuracy of 100 % should not be read as a production estimate. No class-aware metric differences are visible on the clean test (per-class precision/recall all 1.0, `per_class_metrics.csv`).
