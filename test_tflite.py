"""
TensorFlow Lite test skripti
Bu skript TensorFlow və TensorFlow Lite-ın düzgün quraşdırılıb-quraşdırılmadığını yoxlayır.

Qeyd: Windows-da TensorFlow işləməsi üçün Microsoft Visual C++ Redistributable lazımdır.
Endirmək üçün: https://aka.ms/vs/17/release/vc_redist.x64.exe
"""

try:
    import tensorflow as tf
    import numpy as np
    
    print("="*60)
    print("TensorFlow Lite Test / TensorFlow Lite Test")
    print("="*60)
    
    # TensorFlow versiyasını yoxlayırıq / Check TensorFlow version
    print(f"\n✓ TensorFlow versiyası / version: {tf.__version__}")
    
    # Sadə bir model yaradırıq / Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    
    # TFLite-a çeviririk / Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Model ölçüsünü göstəririk / Show model size
    print(f"✓ TensorFlow Lite mövcuddur və işləyir!")
    print(f"✓ TensorFlow Lite is available and working!")
    print(f"  Test model ölçüsü / Test model size: {len(tflite_model) / 1024:.2f} KB")
    
    # İnterpreter test / Test interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # Test input
    input_details = interpreter.get_input_details()
    test_input = np.random.randn(1, 5).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    print(f"✓ TFLite Interpreter test uğurlu / test successful!")
    print(f"\n{'='*60}")
    print("✓ Hazırsınız! / You are ready!")
    print(f"{'='*60}")
    
except ImportError as e:
    print("="*60)
    print("⚠ TensorFlow quraşdırılmayıb / TensorFlow not installed")
    print("="*60)
    print(f"\nXəta / Error: {e}")
    print("\nQuraşdırmaq üçün / To install:")
    print("  pip install tensorflow pillow numpy")
    print("\nWindows üçün əlavə / Additional for Windows:")
    print("  Microsoft Visual C++ Redistributable yükləyin")
    print("  Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    
except Exception as e:
    print("="*60)
    print("✗ Xəta baş verdi / Error occurred")
    print("="*60)
    print(f"Xəta / Error: {e}")
