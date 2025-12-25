# Kod Nümunələri / Code Examples

Bu fayl layihədə istifadə edilən kod nümunələrini və texniki detalları saxlayır.
This file contains code examples and technical details used in the project.

## TensorFlow Lite Optimizasiya Nümunələri / TensorFlow Lite Optimization Examples

### Model Optimizasiyası

```python
# Model ölçüsünü azaltmaq üçün / To reduce model size:
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # float16 istifadə
tflite_model = converter.convert()
```

### Kiçik Neural Network Strukturu

```python
# Optimized generator model (50 latent dim, kiçik layerlər)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(50,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(img_size * img_size * 3, activation='tanh'),
    tf.keras.layers.Reshape((img_size, img_size, 3))
])
```

## PyTorch Optimizasiya Nümunələri / PyTorch Optimization Examples

### Kiçik Generator

```python
class SimpleImageGenerator(nn.Module):
    def __init__(self, latent_dim=50, img_size=64):
        super(SimpleImageGenerator, self).__init__()
        self.img_size = img_size
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, img_size * img_size * 3),
            nn.Tanh()
        )
```

## Model Ölçüləri / Model Sizes

| Framework | Parametr Sayı | Təxmini Ölçü |
|-----------|--------------|--------------|
| TensorFlow Lite (optimized) | ~320K | ~150 KB |
| PyTorch (optimized) | ~320K | ~1.2 MB |

## Gradient Rəsim Yaratma / Gradient Image Generation

### TensorFlow ilə

```python
x = tf.linspace(0.0, 1.0, img_size)
y = tf.linspace(0.0, 1.0, img_size)
xx, yy = tf.meshgrid(x, y)

r = tf.sin(2 * np.pi * xx) * 0.5 + 0.5
g = tf.cos(2 * np.pi * yy) * 0.5 + 0.5
b = tf.sin(2 * np.pi * (xx + yy)) * 0.5 + 0.5

img_tensor = tf.stack([r, g, b], axis=2)
```

### PyTorch ilə

```python
x = torch.linspace(0, 1, img_size)
y = torch.linspace(0, 1, img_size)
xx, yy = torch.meshgrid(x, y, indexing='ij')

r = torch.sin(2 * np.pi * xx) * 0.5 + 0.5
g = torch.cos(2 * np.pi * yy) * 0.5 + 0.5
b = torch.sin(2 * np.pi * (xx + yy)) * 0.5 + 0.5

img_tensor = torch.stack([r, g, b], dim=2)
```

## Test və Debug / Testing and Debugging

### TensorFlow Lite Test

```python
# TFLite interpreter test
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
test_input = np.random.randn(1, 50).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()

output_details = interpreter.get_output_details()
output = interpreter.get_tensor(output_details[0]['index'])
```

## İpuçları / Tips

### Model Ölçüsünü Azaltmaq
1. Latent dimension-u azalt (100 → 50)
2. Layer ölçülərini azalt (512 → 256, 256 → 128)
3. float16 istifadə et
4. TFLite optimizasiya aktivləşdir

### Keyfiyyət vs Ölçü
- Kiçik model = Daha sürətli, daha az yaddaş
- Böyük model = Daha yaxşı keyfiyyət
- Optimal: 50 latent dim, 256 max layer size

### Mobil Cihazlar üçün
- TensorFlow Lite istifadə et
- 64x64 rəsim ölçüsü kifayətdir
- float16 format yaxşı balans verir
