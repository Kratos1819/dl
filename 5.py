"""
RNN sentiment analysis
"""

import numpy as np
import pandas as pd
import tensorflow as tf #type:ignore
import re
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer #type:ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type:ignore
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense #type:ignore
from sklearn.model_selection import train_test_split #type:ignore
from sklearn.preprocessing import LabelEncoder #type:ignore
from tensorflow.keras.utils import to_categorical #type:ignore

# Load dataset
df = pd.read_csv("sentiment_analysis_1.csv")

# Clean text
def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text.lower()).strip()

df["cleaned_text"] = df["text"].apply(clean_text)

# Tokenization & Padding
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["cleaned_text"])
X = pad_sequences(tokenizer.texts_to_sequences(df["cleaned_text"]), maxlen=200)

# Encode Labels
df["encoded_label"] = LabelEncoder().fit_transform(df["label"])
y = to_categorical(df["encoded_label"], num_classes=3)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build & Train Model
model = Sequential([
    Embedding(10000, 128, input_length=200),
    SimpleRNN(64),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend()
plt.show()

# Encode Labels (store the fitted label_encoder instance)
label_encoder = LabelEncoder()
df["encoded_label"] = label_encoder.fit_transform(df["label"])
y = to_categorical(df["encoded_label"], num_classes=3)

# Sentiment Prediction (use the already fitted label_encoder)
def predict_sentiment(text):
    seq = pad_sequences(tokenizer.texts_to_sequences([clean_text(text)]), maxlen=200)
    prediction = model.predict(seq)[0]
    return label_encoder.inverse_transform([np.argmax(prediction)])[0]  # Use the fitted label_encoder

print(predict_sentiment("The product was amazing! I loved it."))
print(predict_sentiment("It was okay, nothing special."))
print(predict_sentiment("Very poor craftsmanship, fell apart."))