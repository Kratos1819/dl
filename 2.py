
"""
Deep Neural Networks (OCR Letter Recognition Dataset)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential# type: ignore 
from tensorflow.keras.layers import Dense, Dropout#type:ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical#type:ignore

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# One-hot encode the labels
y = to_categorical(y, num_classes=3)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # Output layer with 3 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=8,
                    validation_split=0.2,
                    verbose=1)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate predictions
predictions = model.predict(X_test)
y_pred = predictions.argmax(axis=1)
y_true = y_test.argmax(axis=1)

# Classification report
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=data.target_names))