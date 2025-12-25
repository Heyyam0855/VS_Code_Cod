#!/usr/bin/env python
"""Check if TensorFlow and PyTorch are installed."""

print("Checking installed packages...\n")

# Check TensorFlow
try:
    import tensorflow as tf
    print(f"✓ TensorFlow installed: version {tf.__version__}")
    print(f"  - GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
except ImportError:
    print("✗ TensorFlow NOT installed")

# Check PyTorch
try:
    import torch
    print(f"\n✓ PyTorch installed: version {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("\n✗ PyTorch NOT installed")

# Check torchvision
try:
    import torchvision
    print(f"\n✓ Torchvision installed: version {torchvision.__version__}")
except ImportError:
    print("\n✗ Torchvision NOT installed")
