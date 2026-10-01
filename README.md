# English Sentiment Analysis using a Simple RNN

A beginner-friendly Natural Language Processing (NLP) mini-project built with Python, TensorFlow/Keras, Pandas, NumPy, and Scikit-Learn.

This project trains a Simple Recurrent Neural Network (RNN) on the IMDB movie review dataset to classify English text as **Positive** or **Negative** sentiment, and features both batch testing and an interactive command-line mode for real-time predictions.

---

## 📌 Project Overview

- **Dataset**: IMDB Movie Reviews (subset of 10,000 balanced reviews for fast CPU training)
- **Model**: Sequential Neural Network with Embedding + SimpleRNN + Dense layer (Sigmoid)
- **Vocabulary Size**: 5,000 most frequent words
- **Sequence Length**: 100 words per review (pre-padded / post-truncated)
- **Frameworks**: TensorFlow / Keras, Scikit-learn, Pandas, NumPy

---

## 🏗️ Model Architecture

| Layer | Type | Output Shape | Parameters | Description |
|---|---|---|---|---|
| 1 | `Embedding` | `(None, 100, 32)` | 160,000 | Maps word indices (5,000 vocab) to 32-dimensional dense vectors |
| 2 | `SimpleRNN` | `(None, 32)` | 2,080 | Recurrent layer processing temporal sequences with 32 hidden units |
| 3 | `Dense` | `(None, 1)` | 33 | Fully connected layer with Sigmoid activation for binary classification |

- **Loss Function**: `binary_crossentropy`
- **Optimizer**: `adam`
- **Metric**: `accuracy`

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/anandpkt/English-Sentiment-Analysis.git
cd English-Sentiment-Analysis
pip install -r requirements.txt
```

### 2. Run the Script

Execute the main script:

```bash
python sentiment_analysis.py
```

The script will automatically:
1. Download and decode the IMDB dataset.
2. Clean and preprocess review texts (lowercasing, cleaning HTML tags & punctuation).
3. Tokenize and pad sequences.
4. Split data into train and test sets (80/20 stratified split).
5. Build, compile, and train the SimpleRNN model for 5 epochs.
6. Evaluate test accuracy, classification report, and confusion matrix.
7. Run 5 sample test sentences.
8. Launch an interactive mode where you can test your own custom sentences!

---

## 📊 Sample Output

### Model Evaluation:
```text
========== RESULTS ==========
Training accuracy: ~88-92%
Test accuracy    : ~78-83%
Test loss        : ~0.40 - 0.45

Classification Report:
              precision    recall  f1-score   support
    Negative       0.82      0.80      0.81      1000
    Positive       0.81      0.83      0.82      1000
    accuracy                           0.81      2000
```

### Example Predictions:
```text
Input: I really enjoyed this movie
Output: Positive (91% confident)

Input: This movie was boring
Output: Negative (88% confident)

Input: The acting was brilliant and the story was wonderful
Output: Positive (96% confident)
```

---

## 💻 Interactive Mode

When the script finishes training, you can type your own sentences directly into the console:

```text
========== TRY YOUR OWN SENTENCE ==========
Type a sentence and press Enter. Type 'quit' to exit.

Enter a sentence: The cinematography was stunning but the plot was dragged out.
Sentiment: Negative
Confidence: 62%

Enter a sentence: An absolute masterpiece!
Sentiment: Positive
Confidence: 94%
```

---

## 📁 Repository Structure

```
English-Sentiment-Analysis/
├── .gitignore               # Git ignore patterns
├── README.md                # Project documentation and guide
├── requirements.txt         # Python dependencies
└── sentiment_analysis.py    # Complete training, evaluation & inference script
```

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
