# 1D-CNN Sentiment Analysis

A deep learning sentiment analysis project built on the IMDb movie review dataset. This repository focuses on a 1D Convolutional Neural Network (CNN) for binary sentiment classification, with architecture choices inspired by classic text CNN models for sentence-level sentiment classification.

The project was designed as part of a comparative study of multiple neural network architectures, including MLP, 1D-CNN, BiLSTM, and DistilBERT. This repository contains the 1D-CNN implementation and evaluation workflow.

## Overview

Sentiment analysis is the task of determining whether a given text expresses a positive or negative opinion. In this project, we classify movie reviews from the IMDb dataset as either:

- Positive
- Negative

The model treats reviews as sequences of tokens and uses a CNN over word embeddings to capture local n-gram patterns such as trigrams, four-grams, and five-grams.

## Project Goals

- Build a strong text-classification baseline using a 1D CNN
- Prepare and preprocess IMDb movie reviews
- Train and validate a sentiment classifier
- Evaluate model performance using standard metrics
- Save training history, predictions, and ROC/Confusion Matrix outputs

## Model Architecture

The implemented model uses:

- Tokenized review sequences
- Embedding layer
- Spatial dropout regularization
- Three parallel 1D convolution branches with kernel sizes 3, 4, and 5
- Global max pooling after each convolution branch
- Concatenation of pooled features
- Dense classification head with dropout and L2 regularization
- Sigmoid output for binary classification

This architecture follows the idea that different convolution kernel sizes capture different local context lengths in the review text.

## Dataset

The project uses the IMDb movie review dataset from Stanford AI Lab:

- Dataset: ACL IMDb
- Task: Binary sentiment classification
- Classes: 2 (Positive / Negative)
- Train examples: 20,000
- Validation examples: 5,000
- Test examples: 25,000

The notebook automatically downloads the dataset if it is not already present.

## Repository Contents

- `CNN.ipynb` — complete notebook containing data loading, preprocessing, model training, evaluation, plotting, and error analysis
- `README.md` — project overview and usage instructions

## Requirements

This project is implemented with Python and TensorFlow.

Required packages:

- Python 3.9+
- TensorFlow 2.x
- scikit-learn
- NumPy
- matplotlib
- seaborn

Install dependencies:

```bash
pip install -q tensorflow scikit-learn matplotlib seaborn
```

## Running the Project

### Option 1: Google Colab

Open the notebook in Google Colab and run all cells sequentially.

The notebook is configured to:

- mount Google Drive
- create local folders for datasets and outputs
- download the IMDb dataset
- load a shared tokenizer
- train the CNN model
- save training metrics and plots

### Option 2: Local Jupyter Notebook

1. Clone the repository
2. Open `CNN.ipynb` in Jupyter or VS Code Notebook support
3. Run the cells in order
4. Make sure the required directories and paths match your environment

## Training and Evaluation

The notebook includes:

- Model construction and summary
- Early stopping and learning-rate reduction callbacks
- Validation and test evaluation
- Accuracy, precision, recall, F1-score, and ROC-AUC metrics
- Confusion matrix and ROC visualizations
- Error analysis of misclassified samples

## Results

The model achieved strong results on the validation and test sets.

| Split | Accuracy | Precision | Recall | F1-Score | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.8850 | 0.8953 | 0.8720 | 0.8835 | 0.9545 |
| Test | 0.8827 | 0.8946 | 0.8676 | 0.8809 | 0.9515 |

These results show a reliable baseline for sentiment classification using a 1D CNN architecture.

## Notes

- The notebook expects a Drive-based workflow in the original project setup, including a shared tokenizer path.
- If you are running this outside the original environment, you may need to adjust file paths and dataset loading logic.
- The repository is notebook-centric and is best used as an educational or experimentation project rather than a library/package.

## Future Improvements

Possible next steps for this project include:

- Comparing the 1D CNN with BiLSTM and DistilBERT baselines
- Hyperparameter tuning
- Adding cross-validation
- Saving model artifacts for reproducible inference
- Converting the notebook into a reusable Python training pipeline

## License

No explicit license file is included in the repository. If you plan to reuse or distribute this project, check whether the dataset or project structure requires attribution or specific licensing terms.

## Acknowledgements

- IMDb dataset by Stanford AI Lab
- TensorFlow and Keras for deep learning implementation
- scikit-learn for evaluation metrics

