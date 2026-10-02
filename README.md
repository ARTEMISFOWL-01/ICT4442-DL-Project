# Sentiment Analysis on IMDb Reviews: A Four-Model Comparison

A deep learning project comparing four neural-network architectures for binary sentiment classification on the IMDb movie review dataset.

This project explores how model choice affects performance on the same data, while keeping the preprocessing, data split, and evaluation protocol consistent across all models.

## Project Overview

The task is to classify movie reviews as either:

- Positive
- Negative

The dataset is the IMDb Large Movie Review Dataset, containing 50,000 reviews. We use the standard binary sentiment setup, with the 25k test set held out for final evaluation.

We compare the following models:

| Model | Type | Description |
| --- | --- | --- |
| MLP | Feed-forward baseline | Simple embedding-based model with global pooling |
| 1D-CNN | Convolutional neural network | Multi-kernel text CNN for local n-gram learning |
| BiLSTM | Recurrent neural network | Bidirectional sequence model for context capture |
| DistilBERT | Transformer | Fine-tuned pretrained language model |

The goal is not only to maximize accuracy, but also to understand how architecture impacts learning, generalization, and efficiency.

## Why This Project Matters

Different architectures make different assumptions about the input:

- MLP treats reviews as pooled bag-of-words style patterns
- 1D-CNN detects local word patterns and n-grams
- BiLSTM processes reviews as sequences and captures longer dependencies
- DistilBERT leverages pretrained language knowledge and contextualized embeddings

Because all models are trained and evaluated under the same conditions, the results are directly comparable.

## Repository Structure

```text
.
├── MLP.ipynb           # MLP model training and evaluation
├── CNN.ipynb           # 1D-CNN model training and evaluation
├── Bilstm.ipynb        # BiLSTM model training and evaluation
├── DistillBert.ipynb   # DistilBERT fine-tuning and evaluation
├── README.md           # Project documentation
├── figures/            # Plots and visual results (if generated locally)
├── interim_report.tex   # Interim report source
├── synopsis.tex        # Synopsis source
├── synopsis.pdf        # Compiled synopsis PDF
├── Details.txt         # Project/team information and dataset details
└── .gitignore
```

## Dataset

The project uses the IMDb dataset from Stanford AI Lab:

- Dataset: IMDb Large Movie Review Dataset
- Task: Binary sentiment classification
- Reviews: 50,000 total
- Balanced classes: positive and negative
- Train/Val/Test split: 20k train, 5k validation, 25k test

The validation set is used for early stopping and model selection, while the test set remains untouched until the final model comparison.

## Preprocessing

The non-transformer models share a common preprocessing pipeline:

- Tokenizer trained only on training data
- Vocabulary size: 20,000
- Maximum sequence length: 512
- Post-padding and post-truncation
- Shared preprocessing across MLP, CNN, and BiLSTM

For DistilBERT, the model uses the Hugging Face tokenizer based on distilbert-base-uncased and a shorter maximum sequence length due to resource constraints.

## Models at a Glance

### MLP

- Embedding layer followed by spatial dropout
- Global average pooling
- Dense hidden layer and sigmoid output
- Simple baseline with low computational cost
- Useful to quantify how much sequence structure matters

### 1D-CNN

- Embedding layer
- Spatial dropout regularization
- Three parallel convolution branches with kernel sizes 3, 4, and 5
- Global max pooling and feature concatenation
- Dense classification head
- Strong baseline for local text pattern learning

### BiLSTM

- Embedding layer with masking
- Spatial dropout
- Bidirectional LSTM encoder
- Dense classification layer
- Captures temporal context and long-range dependencies in sentences

### DistilBERT

- Fine-tuned DistilBERT backbone
- Classification head added on top
- Pretrained contextual representations
- Strongest performing model in the comparison

## Quick Results

The following results are based on the final held-out test set:

| Model | Test Accuracy | Test F1 | Test AUC |
| --- | ---: | ---: | ---: |
| MLP | 88.27% | 0.8799 | 0.9495 |
| 1D-CNN | 88.27% | 0.8809 | 0.9515 |
| BiLSTM | 80.62% | 0.8211 | 0.8786 |
| DistilBERT | 90.71% | 0.9069 | 0.9691 |

### Interpretation

- DistilBERT performs best, as expected, because it leverages pretrained language understanding.
- MLP and 1D-CNN are competitive, despite being much simpler and cheaper to train.
- BiLSTM underperforms in this setup, suggesting that the chosen configuration or training setup was less suitable for this dataset.
- The performance gap between classic models and the transformer is meaningful, but not overwhelmingly large in this project.

## How to Run

Each notebook is self-contained and designed to run in a Google Colab environment. The notebooks handle:

- dataset download
- preprocessing
- model creation
- training and validation
- metrics computation
- confusion matrix and ROC visualization

### Requirements

For the TensorFlow-based notebooks:

- Python 3.x
- TensorFlow 2.x
- NumPy
- scikit-learn
- matplotlib
- seaborn

For DistilBERT:

- PyTorch
- Hugging Face Transformers
- sentencepiece / tokenizer dependencies as required

### Typical Workflow

1. Open any notebook in Jupyter or Google Colab.
2. Ensure the required dependencies are installed.
3. Run all cells in order.
4. Observe training logs, accuracy curves, and evaluation plots.
5. Compare metrics across the four models.

## Team

| Name | Register Number | Model |
| --- | --- | --- |
| D. Vishwatej | 230953220 | DistilBERT |
| Abhay Pratap Singh | 230911522 | 1D-CNN |
| Rallapalli Dheeraj Chowdary | 230911530 | BiLSTM |
| Y Kedarnath Chowdary | 230911176 | MLP |

## References

1. Maas, Andrew, et al. "Learning Word Vectors for Sentiment Analysis." ACL, 2011.
2. Kim, Yoon. "Convolutional Neural Networks for Sentence Classification." EMNLP, 2014.
3. Hochreiter, Sepp, and Jürgen Schmidhuber. "Long Short-Term Memory." Neural Computation, 1997.
4. Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL, 2019.
5. Sanh, Victor, et al. "DistilBERT, a distilled version of BERT." NeurIPS Workshop, 2019.

## Conclusion

This repository demonstrates that model architecture matters for sentiment analysis, but the value of deeper and more complex models depends on the problem setup, dataset size, and training constraints.

The comparison shows that classic architectures such as MLP and 1D-CNN can still perform very competitively, while transformer-based methods like DistilBERT provide the strongest overall performance.

