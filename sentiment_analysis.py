"""
English Sentiment Analysis using a Simple RNN
Beginner NLP mini project (Python, Pandas, NumPy, TensorFlow/Keras, Scikit-learn)
"""

import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------------------------------------
# Settings (small values so it runs fast on a normal CPU)
# ----------------------------------------------------------
SEED = 42
NUM_REVIEWS = 10000   # how many reviews we use
VOCAB_SIZE = 5000     # keep only the 5000 most common words
MAX_LEN = 100         # every review becomes exactly 100 words long
EMBED_DIM = 32        # each word becomes a vector of 32 numbers
RNN_UNITS = 32        # size of the RNN's "memory"
EPOCHS = 5
BATCH_SIZE = 64

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ----------------------------------------------------------
# 1. DATASET LOADING
# ----------------------------------------------------------
def load_dataset():
    """Load the IMDB reviews and convert them back to readable text."""
    (x_train, y_train), (x_test, y_test) = imdb.load_data()

    # Keras stores IMDB reviews as numbers, so we convert them back to words.
    word_index = imdb.get_word_index()
    index_to_word = {index + 3: word for word, index in word_index.items()}
    # Indexes 0-3 are special markers (padding, start, unknown) - we skip them.

    def decode(review):
        return " ".join(index_to_word.get(i, "") for i in review if i >= 4)

    # Combine the original train and test parts, then pick a random sample.
    all_reviews = np.concatenate([x_train, x_test])
    all_labels = np.concatenate([y_train, y_test])
    chosen = np.random.choice(len(all_reviews), NUM_REVIEWS, replace=False)

    df = pd.DataFrame({
        "review": [decode(all_reviews[i]) for i in chosen],
        "label": all_labels[chosen],      # 1 = positive, 0 = negative
    })
    return df


# ----------------------------------------------------------
# 2. BASIC TEXT PREPROCESSING
# ----------------------------------------------------------
def clean_text(text):
    """Lowercase the text and remove everything except letters and spaces."""
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)     # remove HTML line breaks
    text = re.sub(r"[^a-z' ]", " ", text)      # keep only letters, ' and spaces
    text = re.sub(r"\s+", " ", text).strip()   # remove extra spaces
    return text


print("Loading dataset (first run downloads it, please wait)...")
df = load_dataset()
df["review"] = df["review"].apply(clean_text)
print("Dataset size:", df.shape)
print(df["label"].value_counts().rename({1: "Positive", 0: "Negative"}))
print("\nSample review:", df["review"].iloc[0][:150], "...")


# ----------------------------------------------------------
# 3. TOKENIZATION  (words -> numbers)
# ----------------------------------------------------------
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df["review"])
sequences = tokenizer.texts_to_sequences(df["review"])


# ----------------------------------------------------------
# 4. SEQUENCE PADDING  (make all reviews the same length)
# ----------------------------------------------------------
X = pad_sequences(sequences, maxlen=MAX_LEN, padding="pre", truncating="post")
y = df["label"].values


# ----------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ----------------------------------------------------------
# 6. RNN MODEL CREATION
# ----------------------------------------------------------
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM),
    SimpleRNN(RNN_UNITS),
    Dense(1, activation="sigmoid"),
])
model.build(input_shape=(None, MAX_LEN))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# ----------------------------------------------------------
# 7. MODEL TRAINING
# ----------------------------------------------------------
print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1,
)


# ----------------------------------------------------------
# 8. MODEL EVALUATION
# ----------------------------------------------------------
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\n========== RESULTS ==========")
print(f"Training accuracy: {train_acc * 100:.2f}%")
print(f"Test accuracy    : {test_acc * 100:.2f}%")
print(f"Test loss        : {test_loss:.4f}")

# Convert probabilities (0 to 1) into class predictions (0 or 1)
y_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int).ravel()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

print("Confusion Matrix (rows = actual, columns = predicted):")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(
    cm,
    index=["Actual Negative", "Actual Positive"],
    columns=["Pred Negative", "Pred Positive"],
))


# ----------------------------------------------------------
# 9. PREDICTION FUNCTION
# ----------------------------------------------------------
def predict_sentiment(sentence):
    """Return ('Positive' or 'Negative', confidence in percent)."""
    cleaned = clean_text(sentence)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="pre", truncating="post")
    prob = float(model.predict(padded, verbose=0)[0][0])   # probability of Positive

    if prob >= 0.5:
        return "Positive", prob * 100
    return "Negative", (1 - prob) * 100


# ----------------------------------------------------------
# 10. TESTING WITH 5 EXAMPLE SENTENCES
# ----------------------------------------------------------
print("\n========== TEST EXAMPLES ==========")
examples = [
    "I really enjoyed this movie",
    "This movie was boring",
    "The acting was brilliant and the story was wonderful",
    "What a waste of time, terrible plot and bad acting",
    "It was a great film and I loved every minute",
]
for sentence in examples:
    label, confidence = predict_sentiment(sentence)
    print(f"Input: {sentence}")
    print(f"Output: {label} ({confidence:.0f}% confident)\n")


# ----------------------------------------------------------
# INTERACTIVE MODE: type your own sentence
# ----------------------------------------------------------
print("========== TRY YOUR OWN SENTENCE ==========")
print("Type a sentence and press Enter. Type 'quit' to exit.\n")
while True:
    user_text = input("Enter a sentence: ").strip()
    if user_text.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break
    if not user_text:
        continue
    label, confidence = predict_sentiment(user_text)
    print(f"Sentiment: {label}")
    print(f"Confidence: {confidence:.0f}%\n")
