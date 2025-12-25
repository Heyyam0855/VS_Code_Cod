# TensorFlow və PyTorch Quraşdırma Təlimatı

## Quraşdırma əmrləri

### 1. TensorFlow quraşdırma:
```bash
pip install tensorflow
```

### 2. PyTorch və torchvision quraşdırma:
```bash
pip install torch torchvision
```

### 3. Və ya bütün paketləri bir əmrlə:
```bash
pip install -r requirements.txt
```

## Yoxlama əmri

Quraşdırmanı yoxlamaq üçün:
```bash
python check_packages.py
```

## Problem varsa

Əgər quraşdırma dayandırılırsa (ERROR: Operation cancelled by user), aşağıdakılardan birini sınayın:

### Variant 1: Bir-bir quraşdırma
```bash
# Əvvəlcə PyTorch
pip install torch

# Sonra TensorFlow
pip install tensorflow

# Sonra torchvision
pip install torchvision
```

### Variant 2: İnteraktiv rejimi söndürmə
```bash
pip install --no-input tensorflow torch torchvision
```

### Variant 3: Timeout artırma
```bash
pip install --timeout 1000 tensorflow torch torchvision
```

## Microsoft Visual C++ Redistributable

Əgər bu xəta alırsınızsa:
```
Microsoft Visual C++ Redistributable is not installed
```

Onu buradan yükləyin:
https://aka.ms/vs/17/release/vc_redist.x64.exe

## Test skripti

`check_packages.py` faylını işə salın:
```bash
python check_packages.py
```

Aşağıdakı kimi cavab görmüşsünüzsə, hər şey qaydasındadır:
```
✓ TensorFlow installed: version 2.X.X
✓ PyTorch installed: version 2.X.X
✓ Torchvision installed: version 0.X.X
```

## GPU dəstəyi

GPU istifadə etmək istəyirsinizsə:

### PyTorch CUDA ilə:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### TensorFlow GPU:
TensorFlow 2.x CUDA-nı avtomatik tanıyır (əgər quraşdırılıbsa).
