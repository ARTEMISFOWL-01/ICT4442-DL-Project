# Sentiment Analysis on IMDb Reviews — A Four-Model Comparison

**Course:** ICT 4442 — Deep Learning  
**Institution:** School of Computer Engineering, MIT Manipal (MAHE)

---

## What This Project Is About

We picked a straightforward question: *how much does architecture choice matter for sentiment classification?*

To find out, we trained four different deep learning models on the same dataset (IMDb movie reviews, 50k reviews, binary positive/negative) using the same train/val/test split and the same evaluation metrics. The four models are:

| Model | Type | Who Built It |
|-------|------|--------------|
| MLP | Feed-forward baseline (bag of averaged embeddings) | Y Kedarnath Chowdary |
| 1D-CNN | Multi-kernel convolutional (Kim 2014 style) | Abhay Pratap Singh |
| BiLSTM | Bidirectional recurrent | Rallapalli Dheeraj Chowdary |
| DistilBERT | Fine-tuned pretrained Transformer | D. Vishwatej |

The point isn't just to get a number — it's to understand *why* different architectures give different results on the same data.

## Results (Quick Look)

| Model | Test Accuracy | Test F1 | Test AUC |
|-------|:---:|:---:|:---:|
| MLP | 88.27% | 0.8799 | 0.9495 |
| 1D-CNN | 88.27% | 0.8809 | 0.9515 |
| BiLSTM | 80.62% | 0.8211 | 0.8786 |
| DistilBERT | **90.71%** | **0.9069** | **0.9691** |

DistilBERT wins, but the interesting part is that the simple MLP and CNN are not far behind — and they're way cheaper to train. The BiLSTM struggled with generalization on this particular setup (more on that in the report).

## Repo Structure

```
├── MLP.ipynb           # MLP model — training, eval, plots
├── CNN.ipynb           # 1D-CNN model
├── Bilstm.ipynb        # BiLSTM model
├── DistillBert.ipynb   # DistilBERT fine-tuning
├── figures/            # All plots (training curves, confusion matrices, ROC)
├── interim_report.tex  # Part B interim report (LaTeX source)
├── synopsis.tex        # Synopsis (LaTeX source)
├── synopsis.pdf        # Synopsis (compiled)
└── Details.txt         # Team info and dataset link
```

## Dataset

**IMDb Large Movie Review Dataset** by Maas et al. (2011)  
50,000 reviews — 25k train, 25k test — balanced 50/50 positive/negative.

Download: https://ai.stanford.edu/~amaas/data/sentiment/

We split the 25k training set into 20k train + 5k validation (stratified, `random_state=42`). The 25k test set is kept completely separate — no peeking.

## Preprocessing

All non-Transformer models share the same Keras tokenizer (fitted on train data only, saved to `shared/tokenizer.pkl`):
- Vocab size: 20,000
- Max sequence length: 512 tokens (covers ~92% of reviews)
- Post-padding, post-truncation

DistilBERT uses its own WordPiece tokenizer (`distilbert-base-uncased`) with max length 256 (GPU memory constraint on a T4).

## Models at a Glance

**MLP** — Embedding → SpatialDropout1D(0.2) → GlobalAveragePooling → Dense(64) → Sigmoid. Dead simple, no sequence info. ~2.6M params. Adam, lr=1e-3. 13 epochs, early stopped at epoch 10.

**1D-CNN** — Embedding → SpatialDropout1D(0.3) → three parallel Conv1D branches (kernel sizes 3, 4, 5, 64 filters each) → GlobalMaxPool → concat → Dense(64) → Sigmoid. ~2.7M params. Adam, lr=1e-3. 6 epochs, early stopped at epoch 3.

**BiLSTM** — Embedding (mask_zero) → SpatialDropout1D(0.3) → Bidirectional LSTM(64) → Dense(64) → Sigmoid. ~2.7M params. Adam, lr=5e-4. 5 epochs, early stopped at epoch 2. Slow to train (~11 min/epoch on CPU).

**DistilBERT** — `distilbert-base-uncased` + classification head, full fine-tuning. ~67M params. AdamW, lr=2e-5, linear warmup (10%), gradient clipping. 4 epochs, early stopped at epoch 2. Trained on T4 GPU.

## How to Run

Each notebook is self-contained and was run on Google Colab. Open any `.ipynb`, connect to a runtime, and run all cells. The notebooks handle dataset download, preprocessing, training, and evaluation.

**Requirements** (auto-installed in the notebooks):
- TensorFlow 2.x (MLP, CNN, BiLSTM)
- PyTorch + HuggingFace Transformers (DistilBERT)
- scikit-learn, matplotlib, seaborn

## Team

| Name | Reg. No. | Model |
|------|----------|-------|
| D. Vishwatej | 230953220 | DistilBERT |
| Abhay Pratap Singh | 230911522 | 1D-CNN |
| Rallapalli Dheeraj Chowdary | 230911530 | BiLSTM |
| Y Kedarnath Chowdary | 230911176 | MLP |

## References

1. Maas et al., "Learning Word Vectors for Sentiment Analysis," ACL 2011 — [paper](https://aclanthology.org/P11-1015/)
2. Kim, "Convolutional Neural Networks for Sentence Classification," EMNLP 2014 — [paper](https://aclanthology.org/D14-1181/)
3. Hochreiter & Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997 — [paper](https://doi.org/10.1162/neco.1997.9.8.1735)
4. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers," NAACL 2019 — [paper](https://aclanthology.org/N19-1423/)
5. Sanh et al., "DistilBERT," NeurIPS Workshop 2019 — [paper](https://arxiv.org/abs/1910.01108)
