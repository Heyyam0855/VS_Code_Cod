"""
Simple AI Machine - A basic neural network implementation
This module provides a simple machine learning classifier using TensorFlow/Keras
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class AIMachine:
    """
    A simple AI machine learning classifier
    """
    
    def __init__(self, input_dim=20, hidden_units=64):
        """
        Initialize the AI machine
        
        Args:
            input_dim: Number of input features
            hidden_units: Number of units in hidden layer
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self):
        """Build a simple neural network model"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not installed. Please install it using: pip install -r requirements.txt")
            return False
        
        self.model = keras.Sequential([
            keras.layers.Dense(self.hidden_units, activation='relu', input_shape=(self.input_dim,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return True
    
    def train(self, X, y, epochs=10, batch_size=32, validation_split=0.2):
        """
        Train the AI machine
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
        """
        if self.model is None:
            if not self.build_model():
                return None
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model
        history = self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """
        Make predictions using the AI machine
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return (predictions > 0.5).astype(int).flatten()
    
    def evaluate(self, X, y):
        """
        Evaluate the AI machine
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        loss, accuracy = self.model.evaluate(X_scaled, y, verbose=0)
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }


def demo():
    """Run a demonstration of the AI Machine"""
    print("=== AI Machine Demo ===\n")
    
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}\n")
    
    # Create and train the AI machine
    print("Creating AI Machine...")
    ai = AIMachine(input_dim=20, hidden_units=64)
    
    print("Training AI Machine...")
    history = ai.train(X_train, y_train, epochs=20, batch_size=32)
    
    if history is None:
        print("Training failed. Please install TensorFlow.")
        return
    
    print("\nTraining completed!")
    
    # Evaluate the model
    print("\nEvaluating on test set...")
    results = ai.evaluate(X_test, y_test)
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test Loss: {results['loss']:.4f}")
    
    # Make some predictions
    print("\nMaking predictions on first 5 test samples...")
    predictions = ai.predict(X_test[:5])
    print(f"Predictions: {predictions}")
    print(f"Actual labels: {y_test[:5]}")


if __name__ == "__main__":
    demo()
