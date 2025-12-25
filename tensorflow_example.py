"""
TensorFlow AI Machine Learning Example
A simple neural network for image classification using MNIST dataset
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def create_model():
    """Create a simple neural network model"""
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

def train_model():
    """Train the model on MNIST dataset"""
    print("TensorFlow AI Machine Learning Example")
    print("=" * 50)
    print(f"TensorFlow version: {tf.__version__}")
    
    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    # Create and compile model
    print("\nCreating model...")
    model = create_model()
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        x_train, y_train,
        epochs=5,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Make predictions
    print("\nMaking predictions on first 5 test samples...")
    predictions = model.predict(x_test[:5])
    for i in range(5):
        predicted_label = np.argmax(predictions[i])
        true_label = y_test[i]
        print(f"Sample {i+1}: Predicted={predicted_label}, Actual={true_label}")
    
    return model, history

if __name__ == "__main__":
    try:
        model, history = train_model()
        print("\n" + "=" * 50)
        print("TensorFlow example completed successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure TensorFlow is installed: pip install tensorflow")
