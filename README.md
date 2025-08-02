# Personal Health Mentions Classification: LSTM vs Bi-LSTM

## Project Overview
This project focuses on classifying tweets into two categories:
- **Personal Health Mentions** (tweets that mention personal health conditions)
- **Non-Personal Health Mentions** (tweets unrelated to personal health)

Two deep learning models are implemented and compared for this text classification task:
- **LSTM (Long Short-Term Memory)**
- **Bi-LSTM (Bidirectional LSTM)**

The models are trained and evaluated on a dataset of health-related tweets to identify personal health mentions with emphasis on handling class imbalance.

---

## Dataset
- **Training samples:** 9,991 tweets
- **Test samples:** 3,331 tweets
- Class distribution in test data:
  - Non-Personal Health: 2,364 (71%)
  - Personal Health: 967 (29%)
- Source files: `phm_train.csv` and `phm_test.csv`

---

## Technical Specifications
- Framework: TensorFlow / Keras
- Optimizer: Adam
- Loss function: Binary Crossentropy
- Metrics: Accuracy, Precision, Recall, F1-Score
- Embedding dimension: 100
- LSTM output units: 128
- Batch size: 128
- Training epochs: 10
- Dropout rate: 0.5
- Class weights: {0: 1.0, 1: 2.5} (to address class imbalance)

---

## Data Preprocessing
- HTML tags and non-alphabetic characters removed
- Stopwords removed using NLTK's English stopwords collection
- Text normalized to lowercase
- Tweets tokenized and padded/truncated to fixed length (max length ~10 words)

---

## Model Architectures

### LSTM Model
- Embedding layer (size = vocabulary size x 100)
- LSTM layer with 128 output units
- Dense output layer with sigmoid activation
- Dropout regularization (0.5)

### Bi-LSTM Model
- Embedding layer (size = vocabulary size x 100)
- Bidirectional LSTM layer (forward and backward LSTMs with 128 units each, output concatenated)
- Dense output layer with sigmoid activation
- Dropout regularization (0.5)

---

## Training
- Both models trained for 10 epochs with early stopping on validation loss
- Class imbalance handled via class weights (2.5× penalty for misclassifying personal health mentions)
- Best models saved during training checkpoints








