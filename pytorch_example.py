"""
PyTorch AI Machine Learning Example
A simple neural network for image classification using MNIST dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

class NeuralNetwork(nn.Module):
    """Simple neural network model"""
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_epoch(model, train_loader, loss_fn, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def test(model, test_loader, loss_fn, device):
    """Evaluate the model"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += y.size(0)
    
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def train_model():
    """Train the model on MNIST dataset"""
    print("PyTorch AI Machine Learning Example")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
    # Normalize using MNIST dataset mean (0.1307) and std (0.3081)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = NeuralNetwork().to(device)
    
    print("\nModel Architecture:")
    print(model)
    
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Train the model
    print("\nTraining model...")
    epochs = 5
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = test(model, test_loader, loss_fn, device)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Make predictions
    print("\nMaking predictions on first 5 test samples...")
    model.eval()
    with torch.no_grad():
        # Get first batch from test loader
        for X, y in test_loader:
            X_sample = X[:5].to(device)
            y_sample = y[:5]
            predictions = model(X_sample)
            predicted_labels = predictions.argmax(1)
            
            for i in range(5):
                print(f"Sample {i+1}: Predicted={predicted_labels[i].item()}, Actual={y_sample[i].item()}")
            break
    
    return model

if __name__ == "__main__":
    try:
        model = train_model()
        print("\n" + "=" * 50)
        print("PyTorch example completed successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure PyTorch is installed: pip install torch torchvision")
