# Rəsim Generatorları / Image Generators

PyTorch və TensorFlow Lite ilə sadə rəsim yaratma nümunələri.
Simple image generation examples with PyTorch and TensorFlow Lite.

## Quraşdırma / Installation

```bash
pip install -r requirements.txt
```

## İstifadə / Usage

### PyTorch ilə / With PyTorch

```bash
python pytorch_image_generator.py
```

Bu kod 2 növ rəsim yaradır:
This code generates 2 types of images:
- `pytorch_generated.png` - Neural network ilə yaradılmış (64x64)
- `pytorch_gradient.png` - Gradient rəsim (256x256)

### TensorFlow Lite ilə / With TensorFlow Lite

```bash
python tensorflow_lite_image_generator.py
```

Bu kod 3 fayl yaradır:
This code creates 3 files:
- `tensorflow_gradient.png` - Gradient rəsim (256x256)
- `generator_model.tflite` - TFLite model faylı
- `tflite_generated.png` - TFLite modeli ilə yaradılmış rəsim (64x64)

## Fərqlər / Differences

### PyTorch
- Daha çevik və araşdırma üçün uyğun
- More flexible and suitable for research
- Daha asan debugging
- Easier debugging

### TensorFlow Lite
- Mobil və embedded cihazlar üçün optimizə edilmiş
- Optimized for mobile and embedded devices
- Daha kiçik model ölçüsü
- Smaller model size
- Daha sürətli inference (məlumatın işlənməsi)
- Faster inference

## Nümunələr / Examples

Hər iki kod eyni funksiyaları yerinə yetirir / Both codes perform the same functions:
1. Gradient rəsim yaradırlar (rəng keçidləri) / Create gradient images (color transitions)
2. Neural network istifadə edərək rəsim yaradırlar / Generate images using neural networks
3. Yaradılmış rəsimləri PNG formatında saxlayırlar / Save generated images in PNG format
