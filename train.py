"""
PyTorch Implementation of a Convolutional Autoencoder (CNN Autoencoder) for Image Compression and Reconstruction.
Designed as a beginner-friendly educational guide and foundational research artifact.

Key Architectures:
- Input Image: 1 x 28 x 28 grayscale (784 features)
- Latent Space: 32 dimensions (representing a 24.5x spatial compression ratio)
- Dataset: MNIST (Handwritten digits)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility so the results are consistent every time you run it
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# PHASE 1: SETUP (Hardware Selection)
# =====================================================================
# If a Graphics Processing Unit (GPU) is available (CUDA), use it to speed up training.
# Otherwise, default to the Central Processing Unit (CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using hardware device: {device}\n")

# =====================================================================
# PHASE 2: DATASET LOADING & PREPARATION
# =====================================================================
# We define a pipeline of transformations to apply to the raw dataset:
# 1. transforms.ToTensor(): Converts PIL images (0 to 255 pixel values) to PyTorch Tensors
#    and automatically normalizes the pixel values to a range between 0.0 and 1.0.
transform = transforms.Compose([
    transforms.ToTensor()
])

# Download and load the MNIST training dataset
print("[*] Loading training dataset...")
train_dataset = datasets.MNIST(
    root="./data",          # Subdirectory where raw data will be saved
    train=True,             # Download the training split (60,000 images)
    download=True,          # Automatically download it if not present
    transform=transform     # Apply the ToTensor transformation
)

# Download and load the MNIST testing/evaluation dataset
print("[*] Loading testing dataset...")
test_dataset = datasets.MNIST(
    root="./data",
    train=False,            # Download the test split (10,000 images)
    download=True,
    transform=transform
)

# DataLoaders handle automatic batching, shuffling, and multi-threaded loading.
# Batch size controls how many images the network processes at once.
BATCH_SIZE = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"[*] Dataset loaded successfully!")
print(f"    - Training batches: {len(train_loader)} (total images: {len(train_dataset)})")
print(f"    - Testing batches: {len(test_loader)} (total images: {len(test_dataset)})\n")


# =====================================================================
# PHASE 3: MODEL ARCHITECTURE (CNN Autoencoder)
# =====================================================================
class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(CNNAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # -------------------------------------------------------------
        # ENCODER: Compresses high-dimensional images to a lower dimension
        # Input shape: (Batch Size, 1, 28, 28)
        # -------------------------------------------------------------
        self.encoder = nn.Sequential(
            # Layer 1: Conv2d (Input channels = 1, Output channels = 16)
            # - Kernel size 3x3: scans 3x3 pixel areas to extract low-level features (edges, curves)
            # - Stride 2: moves the kernel by 2 pixels at a time, cutting spatial dimensions in half!
            # - Padding 1: adds a 1-pixel border of zeros around the image to allow scanning edges
            # Math for output width: W_out = floor((28 - 3 + 2*1)/2) + 1 = 14
            # Output tensor shape: (Batch Size, 16, 14, 14)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # Rectified Linear Unit activation introduces non-linearity
            
            # Layer 2: Conv2d (Input channels = 16, Output channels = 32)
            # - Kernel size 3x3, Stride 2, Padding 1: reduces spatial dimensions in half again
            # Math: W_out = floor((14 - 3 + 2*1)/2) + 1 = 7
            # Output tensor shape: (Batch Size, 32, 7, 7)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Flatten Layer: converts the 3D grid of features (32 channels, 7x7 height/width)
            # into a 1D vector of length 32 * 7 * 7 = 1568 features.
            # Output shape: (Batch Size, 1568)
            nn.Flatten(),
            
            # Latent Layer: a dense (linear) layer that projects 1568 features down to
            # our narrow bottleneck, the "latent dimension" of 32.
            # Output shape: (Batch Size, latent_dim)
            nn.Linear(32 * 7 * 7, latent_dim)
        )
        
        # -------------------------------------------------------------
        # DECODER: Reconstructs the original image from the latent space bottleneck
        # Input shape: (Batch Size, latent_dim)
        # -------------------------------------------------------------
        self.decoder = nn.Sequential(
            # Step 1: Project the compressed latent representation (size 32)
            # back up to the flattened size of the last convolutional layer (1568).
            # Output shape: (Batch Size, 1568)
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            
            # Unflatten Layer: Reshapes the 1D vector back to a 3D feature grid of shape (32, 7, 7)
            # Output shape: (Batch Size, 32, 7, 7)
            nn.Unflatten(dim=1, unflattened_size=(32, 7, 7)),
            
            # Layer 1 (Transpose): Reverses Layer 2 of the Encoder
            # - ConvTranspose2d: "upsamples" or expands spatial dimensions using learned filters
            # - Stride 2: doubles the spatial dimensions
            # - output_padding 1: adjusts the final output dimension to precisely match the target
            # Output tensor shape: (Batch Size, 16, 14, 14)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Layer 2 (Transpose): Reverses Layer 1 of the Encoder
            # - Upsamples back to the original image dimensions: (1, 28, 28)
            # - Sigmoid Activation: restricts final outputs to [0.0, 1.0], matching our normalized input pixel values!
            # Output tensor shape: (Batch Size, 1, 28, 28)
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Compress input image `x` into latent bottleneck `z`
        z = self.encoder(x)
        # 2. Decompress latent bottleneck `z` into reconstructed output `x_reconstructed`
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed

# Instantiate the model and load it onto our device (CPU/GPU)
model = CNNAutoencoder(latent_dim=32).to(device)
print("[*] Model Architecture instantiated:")
print(model)

# Demonstration of dimensional change (Beginner-friendly visualization of forward pass)
print("\n[*] Architecture Shape Demonstration:")
sample_input = torch.randn(1, 1, 28, 28).to(device)
with torch.no_grad():
    sample_latent, sample_output = model(sample_input)
    print(f"    - Input image dimension:        {sample_input.shape}")
    print(f"    - Flattened feature dimension:   [1, 1568]")
    print(f"    - Compressed Latent Dimension:   {sample_latent.shape}  <-- BOTTLENECK!")
    print(f"    - Reconstructed output dimension:{sample_output.shape}")
    
    # Calculate and print theoretical metrics
    original_size_bits = 28 * 28 * 32  # 784 floats x 32-bit floats = 25,088 bits
    latent_size_bits = 32 * 32         # 32 floats x 32-bit floats = 1,024 bits
    compression_ratio = original_size_bits / latent_size_bits
    print(f"    - Theoretical Spatial Compression Ratio: {28*28}/{32} = {(28*28)/32:.1f}x")
    print(f"    - Data Reduction: {100 * (1 - 32/(28*28)):.2f}%\n")


# =====================================================================
# PHASE 4: LOSS FUNCTION & OPTIMIZER
# =====================================================================
# Reconstruction Loss: Mean Squared Error (MSE)
# Measures the average squared difference between input pixel values and reconstructed pixel values.
# MSE = (1 / N) * sum((input_i - output_i)^2)
criterion = nn.MSELoss()

# Optimizer: Adam (Adaptive Moment Estimation)
# Updates the weights of our network by tracking learning rates dynamically.
# Learning rate (lr) controls the step size of parameter updates.
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# =====================================================================
# PHASE 5: TRAINING THE MODEL
# =====================================================================
EPOCHS = 5
print(f"[*] Starting Autoencoder training for {EPOCHS} epochs...")

for epoch in range(1, EPOCHS + 1):
    model.train()  # Put the model in training mode
    train_loss = 0.0
    
    # Iterate through batches of training data
    for batch_idx, (images, _) in enumerate(train_loader):
        # Move inputs to the current device (CPU/GPU)
        images = images.to(device)
        
        # 1. Clear gradients from the previous optimization step
        optimizer.zero_grad()
        
        # 2. Forward pass: compute latent features and reconstructed images
        latent_vectors, reconstructions = model(images)
        
        # 3. Calculate reconstruction loss (difference between original and reconstructed images)
        loss = criterion(reconstructions, images)
        
        # 4. Backward pass: compute gradients of the loss with respect to model weights
        loss.backward()
        
        # 5. Optimization step: update model weights using the calculated gradients
        optimizer.step()
        
        # Accumulate training loss
        train_loss += loss.item() * images.size(0)
    
    # Calculate average training loss for this epoch
    average_train_loss = train_loss / len(train_loader.dataset)
    
    # Put the model in evaluation mode to test on unseen data
    model.eval()
    test_loss = 0.0
    with torch.no_grad():  # Turn off gradient tracking to save memory and speed up evaluation
        for test_images, _ in test_loader:
            test_images = test_images.to(device)
            _, test_reconstructions = model(test_images)
            loss = criterion(test_reconstructions, test_images)
            test_loss += loss.item() * test_images.size(0)
            
    average_test_loss = test_loss / len(test_loader.dataset)
    
    print(f"    Epoch [{epoch}/{EPOCHS}] | Train MSE: {average_train_loss:.6f} | Test MSE: {average_test_loss:.6f}")

print("[*] Training completed successfully!\n")


# =====================================================================
# PHASE 6: EVALUATION & VISUALIZATION
# =====================================================================
print("[*] Generating reconstruction visualizations...")
model.eval()

# Retrieve a single batch of testing images (128 samples)
test_iterator = iter(test_loader)
images, labels = next(test_iterator)
images = images.to(device)

# Get the model reconstructions
with torch.no_grad():
    latent_vectors, reconstructions = model(images)

# Convert tensors back to numpy arrays for plotting
images = images.cpu().numpy()
reconstructions = reconstructions.cpu().numpy()

# Use an unseeded random number generator to select 10 different random digits from the batch
# so that you get different digits EVERY time you run the script, while keeping model training deterministic!
rng = np.random.default_rng()
random_indices = rng.choice(len(images), size=10, replace=False)

# Set up matplotlib figure to show 10 original digits and 10 reconstructed digits
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(14, 4))
fig.suptitle("CNN Autoencoder Image Reconstruction (MNIST)\nTop Row: Original Images | Bottom Row: Reconstructed (Compressed to 32 Dimensions)", fontsize=14, fontweight='bold')

for i, idx in enumerate(random_indices):
    # Plot Original Images
    axes[0, i].imshow(images[idx].squeeze(), cmap='gray')
    axes[0, i].get_xaxis().set_visible(False)
    axes[0, i].get_yaxis().set_visible(False)
    if i == 0:
        axes[0, i].set_ylabel("Original", fontsize=12, fontweight='bold')
    
    # Plot Reconstructed Images
    axes[1, i].imshow(reconstructions[idx].squeeze(), cmap='gray')
    axes[1, i].get_xaxis().set_visible(False)
    axes[1, i].get_yaxis().set_visible(False)
    if i == 0:
        axes[1, i].set_ylabel("Reconstructed", fontsize=12, fontweight='bold')

plt.tight_layout()

# Save the visualization to a high-quality PNG file
output_filename = "reconstruction_results.png"
plt.savefig(output_filename, dpi=300)
print(f"[*] Visual comparisons saved successfully as: {os.path.abspath(output_filename)}")

# Summary stats for academic presentation
print("\n" + "="*60)
print("              SUMMARY OF COMPRESSION STATS FOR ACADEMICS")
print("="*60)
print(f"1. Input Resolution:      28 x 28 pixels = 784 dimensions (per channel)")
print(f"2. Latent Bottleneck:      32 floats")
print(f"3. Spatial Compression:    24.5x reduction")
print(f"4. Model MSE Loss:         {average_test_loss:.5f} (on unseen test set)")
print(f"5. Saved Plot:             {output_filename}")
print("="*60 + "\n")
