
"""
Deep Neural Networks on Text Data (IMDB Dataset)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout #type:ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type:ignore
from tensorflow.keras.datasets import imdb #type:ignore

# Load the IMDB dataset
vocab_size = 100  # Use the top 10,000 words in the dataset
max_length = 20  # Max length of each review
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to ensure uniform input shape
x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

# Build the deep neural network model
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),  # Embedding layer
    Bidirectional(LSTM(64, return_sequences=False)),  # Bidirectional LSTM for sequential data
    Dropout(0.3),  # Dropout for regularization
    Dense(64, activation='relu'),  # Dense layer
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(
    x_train, y_train, 
    epochs=1, 
    batch_size=32, 
    validation_split=0.2, 
    verbose=1
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")