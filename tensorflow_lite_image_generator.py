"""
TensorFlow Lite ilə sadə rəsim yaradıcı
Simple image generator using TensorFlow Lite
"""

import tensorflow as tf
import numpy as np
from PIL import Image


def create_simple_generator_model(latent_dim=50, img_size=64):
    """
    Sadə generator modeli yaradır (optimallaşdırılmış)
    Creates simple generator model (optimized)
    
    Args:
        latent_dim: Giriş vektorunun ölçüsü / Input vector size (azaldılıb/reduced)
        img_size: Rəsim ölçüsü / Image size
    
    Returns:
        TensorFlow modeli / TensorFlow model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),  # 128-dən 64-ə azaldılıb / reduced from 128 to 64
        tf.keras.layers.Dense(128, activation='relu'),  # 256-dan 128-ə / from 256 to 128
        tf.keras.layers.Dense(256, activation='relu'),  # 512-dən 256-ya / from 512 to 256
        tf.keras.layers.Dense(img_size * img_size * 3, activation='tanh'),
        tf.keras.layers.Reshape((img_size, img_size, 3))
    ])
    
    return model


def convert_to_tflite(model, model_path='generator_model.tflite'):
    """
    TensorFlow modelini TFLite formatına çevirir
    Converts TensorFlow model to TFLite format
    
    Args:
        model: TensorFlow modeli / TensorFlow model
        model_path: TFLite model faylının yolu / TFLite model file path
    
    Returns:
        TFLite model yolu / TFLite model path
    """
    print("Model TensorFlow Lite formatına çevrilir... / Converting model to TensorFlow Lite...")
    
    # TFLite konverterini yaradırıq / Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Güclü optimizasiya (model ölçüsünü azaldır) / Strong optimization (reduces model size)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]  # float16 istifadə et / use float16
    
    # Çevirmə / Convert
    tflite_model = converter.convert()
    
    # Model faylını saxlayırıq / Save model file
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ TFLite model saxlanıldı: {model_path}")
    print(f"✓ TFLite model saved: {model_path}")
    print(f"  Fayl ölçüsü / File size: {len(tflite_model) / 1024:.2f} KB")
    
    return model_path


def generate_image_tflite(model_path='generator_model.tflite', 
                          output_path='tflite_generated.png', 
                          img_size=64):
    """
    TensorFlow Lite istifadə edərək rəsim yaradır
    Generates image using TensorFlow Lite
    
    Args:
        model_path: TFLite model faylının yolu / TFLite model file path
        output_path: Çıxış faylının yolu / Output file path
        img_size: Rəsim ölçüsü / Image size
    """
    print("\nTensorFlow Lite ilə rəsim yaradılır... / Generating image with TensorFlow Lite...")
    
    # TFLite interpreterini yükləyirik / Load TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Input və output detaylarını əldə edirik / Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Random noise vektoru yaradırıq / Create random noise vector
    z = np.random.randn(1, 50).astype(np.float32)  # 100-dən 50-yə azaldılıb / reduced from 100 to 50
    
    # İnput setləyirik / Set input
    interpreter.set_tensor(input_details[0]['index'], z)
    
    # Modeli işlədirik / Run inference
    interpreter.invoke()
    
    # Nəticəni alırıq / Get output
    generated_img = interpreter.get_tensor(output_details[0]['index'])
    
    # Şəklə çeviririk / Convert to image
    img_array = generated_img[0]
    
    # [-1, 1] aralığından [0, 255] aralığına keçid / Scale from [-1, 1] to [0, 255]
    img_array = ((img_array + 1) * 127.5).astype(np.uint8)
    
    # Şəkli saxlayırıq / Save image
    img = Image.fromarray(img_array)
    img.save(output_path)
    
    print(f"✓ Rəsim saxlanıldı: {output_path}")
    print(f"✓ Image saved: {output_path}")
    print(f"  Ölçü / Size: {img_size}x{img_size}")
    
    return img_array


def create_gradient_image_tensorflow(output_path='tensorflow_gradient.png', img_size=256):
    """
    TensorFlow ilə gradient rəsim yaradır (daha sadə nümunə)
    Creates gradient image with TensorFlow (simpler example)
    
    Args:
        output_path: Çıxış faylının yolu / Output file path
        img_size: Rəsim ölçüsü / Image size
    """
    print("\nTensorFlow ilə gradient rəsim yaradılır... / Creating gradient image with TensorFlow...")
    
    # Koordinat gridləri yaradırıq / Create coordinate grids
    x = tf.linspace(0.0, 1.0, img_size)
    y = tf.linspace(0.0, 1.0, img_size)
    
    # Mesh grid yaradırıq / Create mesh grid
    xx, yy = tf.meshgrid(x, y)
    
    # Rəng kanallarını yaradırıq / Create color channels
    r = tf.sin(2 * np.pi * xx) * 0.5 + 0.5
    g = tf.cos(2 * np.pi * yy) * 0.5 + 0.5
    b = tf.sin(2 * np.pi * (xx + yy)) * 0.5 + 0.5
    
    # RGB şəkil yaradırıq / Create RGB image
    img_tensor = tf.stack([r, g, b], axis=2)
    
    # [0, 255] aralığına keçid / Scale to [0, 255]
    img_array = (img_tensor.numpy() * 255).astype(np.uint8)
    
    # Şəkli saxlayırıq / Save image
    img = Image.fromarray(img_array)
    img.save(output_path)
    
    print(f"✓ Gradient rəsim saxlanıldı: {output_path}")
    print(f"✓ Gradient image saved: {output_path}")
    print(f"  Ölçü / Size: {img_size}x{img_size}")
    
    return img_array


if __name__ == "__main__":
    print("=" * 60)
    print("TensorFlow Lite Rəsim Generatoru / TensorFlow Lite Image Generator")
    print("=" * 60)
    
    # TensorFlow versiyasını yoxlayırıq / Check TensorFlow version
    print(f"\nTensorFlow versiya / version: {tf.__version__}")
    
    # 1. Gradient rəsim yaradırıq / Create gradient image
    create_gradient_image_tensorflow('tensorflow_gradient.png', img_size=256)
    
    # 2. Neural network modeli yaradırıq / Create neural network model
    print("\nNeural network modeli yaradılır... / Creating neural network model...")
    model = create_simple_generator_model(latent_dim=50, img_size=64)  # Optimallaşdırılmış / Optimized
    print("✓ Model yaradıldı / Model created")
    print(f"  Parametrlər / Parameters: {model.count_params()}")
    print(f"  Təxmini ölçü / Estimated size: ~{model.count_params() * 4 / 1024:.2f} KB")
    
    # 3. Modeli TFLite formatına çeviririk / Convert model to TFLite format
    tflite_model_path = convert_to_tflite(model, 'generator_model.tflite')
    
    # 4. TFLite modeli ilə rəsim yaradırıq / Generate image with TFLite model
    generate_image_tflite(tflite_model_path, 'tflite_generated.png', img_size=64)
    
    print("\n" + "=" * 60)
    print("✓ Bütün rəsimlər uğurla yaradıldı!")
    print("✓ All images generated successfully!")
    print("=" * 60)
    print("\nYaradılmış fayllar / Created files:")
    print("  - tensorflow_gradient.png (256x256 gradient rəsim)")
    print("  - generator_model.tflite (TFLite model)")
    print("  - tflite_generated.png (64x64 neural network ilə yaradılmış)")
