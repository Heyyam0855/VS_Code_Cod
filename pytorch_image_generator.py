"""
PyTorch ilə sadə rəsim yaradıcı
Simple image generator using PyTorch
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class SimpleImageGenerator(nn.Module):
    """Sadə rəsim generatoru (optimallaşdırılmış) / Simple image generator (optimized)"""
    
    def __init__(self, latent_dim=50, img_size=64):
        super(SimpleImageGenerator, self).__init__()
        self.img_size = img_size
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),  # 128-dən 64-ə / from 128 to 64
            nn.ReLU(),
            nn.Linear(64, 128),  # 256-dan 128-ə / from 256 to 128
            nn.ReLU(),
            nn.Linear(128, 256),  # 512-dən 256-ya / from 512 to 256
            nn.ReLU(),
            nn.Linear(256, img_size * img_size * 3),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, self.img_size, self.img_size)
        return img


def generate_image_pytorch(output_path='pytorch_generated.png', img_size=64):
    """
    PyTorch istifadə edərək rəsim yaradır
    Generates image using PyTorch
    
    Args:
        output_path: Çıxış faylının yolu / Output file path
        img_size: Rəsim ölçüsü / Image size
    """
    print("PyTorch ilə rəsim yaradılır... / Generating image with PyTorch...")
    
    # Generator modelini yaradırıq / Create generator model
    generator = SimpleImageGenerator(latent_dim=100, img_size=img_size)
    generator.eval()
    
    # Random noise vektoru / Random noise vector
    with torch.no_grad():
        z = torch.randn(1, 50)  # 100-dən 50-yə / from 100 to 50
        generated_img = generator(z)
    
    # Tensoru şəkilə çeviririk / Convert tensor to image
    img_array = generated_img.squeeze(0).permute(1, 2, 0).numpy()
    
    # [-1, 1] aralığından [0, 255] aralığına keçid / Scale from [-1, 1] to [0, 255]
    img_array = ((img_array + 1) * 127.5).astype(np.uint8)
    
    # Şəkli saxlayırıq / Save image
    img = Image.fromarray(img_array)
    img.save(output_path)
    
    print(f"✓ Rəsim saxlanıldı: {output_path}")
    print(f"✓ Image saved: {output_path}")
    print(f"  Ölçü / Size: {img_size}x{img_size}")
    print(f"  Model parametrləri / Model parameters: {sum(p.numel() for p in generator.parameters())}")
    
    return img_array


def create_gradient_image_pytorch(output_path='pytorch_gradient.png', img_size=256):
    """
    PyTorch ilə gradient rəsim yaradır (daha sadə nümunə)
    Creates gradient image with PyTorch (simpler example)
    
    Args:
        output_path: Çıxış faylının yolu / Output file path
        img_size: Rəsim ölçüsü / Image size
    """
    print("\nPyTorch ilə gradient rəsim yaradılır... / Creating gradient image with PyTorch...")
    
    # Koordinat gridləri yaradırıq / Create coordinate grids
    x = torch.linspace(0, 1, img_size)
    y = torch.linspace(0, 1, img_size)
    
    # Mesh grid yaradırıq / Create mesh grid
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Rəng kanallarını yaradırıq / Create color channels
    r = torch.sin(2 * np.pi * xx) * 0.5 + 0.5
    g = torch.cos(2 * np.pi * yy) * 0.5 + 0.5
    b = torch.sin(2 * np.pi * (xx + yy)) * 0.5 + 0.5
    
    # RGB şəkil yaradırıq / Create RGB image
    img_tensor = torch.stack([r, g, b], dim=2)
    
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
    print("PyTorch Rəsim Generatoru / PyTorch Image Generator")
    print("=" * 60)
    
    # PyTorch versiyasını yoxlayırıq / Check PyTorch version
    print(f"\nPyTorch versiya / version: {torch.__version__}")
    
    # 1. Neural network ilə rəsim yaradırıq / Generate image with neural network
    generate_image_pytorch('pytorch_generated.png', img_size=64)
    
    # 2. Gradient rəsim yaradırıq / Create gradient image
    create_gradient_image_pytorch('pytorch_gradient.png', img_size=256)
    
    print("\n" + "=" * 60)
    print("✓ Bütün rəsimlər uğurla yaradıldı!")
    print("✓ All images generated successfully!")
    print("=" * 60)
