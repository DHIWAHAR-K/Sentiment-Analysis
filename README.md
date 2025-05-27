# Sentiment Analysis

This project implements a **complex deep learning model** using **stacked Bidirectional LSTM layers** for **sentiment classification** of multilingual tweets. The dataset combines labeled tweets from three major political parties in India and focuses on analyzing sentiment in the `in` (Hindi/Indic) language.

---

## 📁 Project Structure

| File/Step                | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| Dataset loading          | Reads and merges labeled tweet data from three CSV files.                   |
| Preprocessing            | Cleans text using regex and tokenizes sentences into sequences.             |
| Tokenization             | Fits a Keras tokenizer and pads sequences to max length.                    |
| Model Architecture       | Defines a deep BiLSTM model with dropout, normalization, and dense layers.  |
| Training                 | Trains the model with early stopping on validation accuracy.                |
| Evaluation               | Evaluates accuracy on unseen test data.                                     |
| Inference                | Predicts sentiment (binary) for a custom tweet input.                       |

---

## Dataset Sources

- `PMLN_predicted_tweets.csv`
- `PPP_predicted_tweets.csv`
- `PTI_predicted_tweets.csv`

> All datasets are filtered for the `in` language and non-null preprocessed tweets.

---

## Preprocessing Pipeline

- Remove non-alphabetic characters
- Normalize whitespaces
- Lowercase and strip text
- Add special tokens (`[start]`, `[end]`) *(optional)*
- Tokenize and convert to padded sequences

---

## Model Architecture

A complex BiLSTM model with the following structure:

- `Embedding Layer`: Projects token indices to embedding vectors
- `SpatialDropout1D`: Helps prevent overfitting
- **BiLSTM Block 1**: `LSTM(64)` with batch norm and dropout
- **BiLSTM Block 2**: `LSTM(32)` stacked on top
- `Concatenate`: Merges outputs from both BiLSTM blocks
- **BiLSTM Block 3**: Final `LSTM(16)` for feature refinement
- Fully Connected Layers: `Dense(128)` → `Dropout` → `Dense(64)` → `Dropout`
- Output Layer: `Dense(1, sigmoid)` for binary classification

> Loss: Binary Crossentropy  
> Optimizer: Adam  
> Metric: Accuracy

---

## Training Details

| Parameter        | Value                                     |
|------------------|-------------------------------------------|
| `batch_size`     | 8                                         |
| `epochs`         | 100 (early stopping on `val_accuracy`)    |
| `embedding_dim`  | 32                                        |
| `vocab_size`     | 87554                                     |
| `max_len`        | Calculated from longest tweet in training |

### Early Stopping

Stops training if validation accuracy does not improve after 3 epochs.

---

## 🧾 How to Run

1. **Prepare Dataset**: Place the three `.csv` files in the appropriate path.
2. **Run Script**:
    ```bash
    python sentiment_analysis_bilstm.py
    ```

3. **Predict on Custom Sentence**:
    ```python
    sentence = ["ghatiya insaan ho tum!!"]
    # Output: Sentiment for this tweet is: 1 (Positive) or 0 (Negative)
    ```

---

## Results

Evaluates the trained model on the held-out test set

## Requirements
```bash
pip install tensorflow pandas scikit-learn datasets transformers
```

## Example Prediction
sentence = ["ghatiya insaan ho tum!!"]
# Preprocess + tokenize + pad
output = model.predict(sentence)
# Threshold = 0.5 → Output: FAKE (1) or REAL (0)

## License
This project is open-source and intended for educational and research purposes.
