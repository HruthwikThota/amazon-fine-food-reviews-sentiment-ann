# Amazon Review Sentiment Analysis (ANN)

An end-to-end multi-class sentiment and star-rating (1–5) prediction system built on the **Amazon Fine Food Reviews** dataset using **TF-IDF** feature extraction and a **TensorFlow/Keras Sequential Artificial Neural Network (ANN)**, deployed interactively via **Streamlit Community Cloud**.

- **Live App:** [Amazon Review Sentiment Analysis (ANN)](https://amazon-review-sentiment-ann-x5ssggoa8n6nmtbncfnuq.streamlit.app)
- **GitHub Repository:** [HruthwikThota/amazon-review-sentiment-ann](https://github.com/HruthwikThota/amazon-review-sentiment-ann)

---

## Project Overview

Customer reviews contain non-linear word interactions, subtle intensity cues, and class imbalance across star ratings (`1` to `5`). In this project, a **Sequential Dense ANN** trained on TF-IDF features (`max_features=1500`) was developed and benchmarked against a traditional **TF-IDF + LinearSVC (`C=1`)** baseline:

- **Traditional ML Baseline (`TF-IDF + LinearSVC`):** **72.04%** test accuracy
- **Deep Learning Model (`TF-IDF + Sequential ANN`):** **77.72%** test accuracy (**+5.68 percentage points** improvement across **113,691** test reviews)

---

## Why the ANN Outperformed Traditional ML (`LinearSVC`)

1. **Non-Linear Feature Interactions Across Hidden Layers:** While `LinearSVC` learns a single linear decision boundary in TF-IDF space, stacked `Dense` layers with `ReLU` activations (`128 -> 64 -> 32 -> 5` with `Dropout`) combine co-occurring sentiment cues and modifiers to separate rating classes more effectively.
2. **Full-Scale Training Across 568,454 Reviews:** Training on the full Amazon Fine Food Reviews corpus (`568,454` total rows; `454,763` training split / `113,691` test split) with memory-safe `float32` TF-IDF arrays and mini-batch training (`batch_size=512`) gave the network rich exposure to vocabulary variations without exhausting runtime RAM.
3. **Calibrated Multi-Class Probabilities (`Softmax`):** Unlike a linear margin classifier, the 5-unit `softmax` output layer estimates class probabilities across all five star ratings (`1` to `5`), making it straightforward to map predictions to `Negative` (`1–2`), `Neutral` (`3`), and `Positive` (`4–5`) along with prediction confidence.

---

## Dataset & Preprocessing Pipeline

- **Source Dataset:** Amazon Fine Food Reviews (`Reviews.csv` — **568,454** rows, **10** columns)
- **Target Mapping:** Star ratings (`1–5`) mapped to zero-indexed classes (`0–4`) and grouped for sentiment reporting as:
  - **1–2 Stars:** `Negative`
  - **3 Stars:** `Neutral`
  - **4–5 Stars:** `Positive`
- **Text Cleaning:** Lowercasing, HTML tag removal, URL removal, punctuation stripping, and whitespace normalization.
- **Train / Test Split:** Stratified `80/20` split (`random_state=42`) producing **113,691** test records.
- **Feature Representation:** `TfidfVectorizer(max_features=1500, ngram_range=(1, 1))` fit strictly on training text to prevent data leakage and cast to `float32` for memory efficiency.

---

## Model Architecture & Training

- **Input Layer:** `Dense(128, activation="relu")` with `Dropout(0.3)`
- **Hidden Layer 1:** `Dense(64, activation="relu")` with `Dropout(0.2)`
- **Hidden Layer 2:** `Dense(32, activation="relu")`
- **Output Layer:** `Dense(5, activation="softmax")`
- **Compilation & Training:** `optimizer="adam"`, `loss="sparse_categorical_crossentropy"`, `metrics=["accuracy"]`, trained with `validation_split=0.2` and `batch_size=512`.

---

## Evaluation Results & Key Takeaways

- **Test Accuracy:** **77.72%** (`0.7772`)
- **Test Loss:** `1.0205`

| Rating Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **1 Star** | 0.69 | 0.71 | 0.70 | 10,454 |
| **2 Stars** | 0.51 | 0.46 | 0.48 | 5,954 |
| **3 Stars** | 0.54 | 0.49 | 0.51 | 8,528 |
| **4 Stars** | 0.58 | 0.50 | 0.53 | 16,131 |
| **5 Stars** | 0.87 | 0.91 | 0.89 | 72,624 |
| **Accuracy / Weighted Avg** | **0.77** | **0.78** | **0.77** | **113,691** |

### Observations from Training Curves & Confusion Matrix
- **Strong performance on extreme/high ratings:** 5-star reviews achieve **0.89 F1** and **0.91 recall**, and 1-star reviews achieve **0.70 F1**.
- **Class imbalance & adjacent rating overlap:** Middle ratings (`2`, `3`, and `4` stars) have fewer samples and share overlapping language with adjacent scores (for instance, moderately positive or 4-star phrasing often leans toward 5 stars).
- **Overfitting insight:** Training accuracy rises above `0.92` while validation accuracy plateaus around `0.78` and validation loss begins climbing after the early epochs—highlighting a clear next step to add `EarlyStopping` / `ModelCheckpoint` to lock in the lowest-loss epoch.

---

## Repository Structure

```text
amazon-review-sentiment-ann/
├── Amazon_Review_sentiment_Analysis_ANN_.ipynb  # Training, evaluation, plots & artifact export
├── app (1).py                                   # Streamlit inference web application
├── final_ann_model.keras                        # Saved Keras Sequential ANN model
├── tfidf_vectorizer.pkl                         # Fitted TF-IDF vectorizer
├── requirements.txt                             # Python dependencies for deployment
└── README.md                                    # Project documentation
