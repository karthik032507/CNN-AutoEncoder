"""
Pure NumPy Implementation of a Deep Autoencoder for MNIST Image Compression.
This is a robust fallback solution designed to run out-of-the-box in case the local 
PyTorch installation experiences Windows dynamic link library loading failures (WinError 1114).

Key Features:
- 100% Pure Python & NumPy (Zero PyTorch dependencies).
- Automatically downloads raw MNIST binary files, decompresses them, and parses idx headers.
- Implements a deep fully-connected autoencoder with manual backpropagation and an Adam optimizer in NumPy.
- Architecture: Input (784) -> Hidden (64) -> Latent Bottleneck (32) -> Hidden (64) -> Output (784)
- Outputs a gorgeous original vs. reconstructed image comparison to 'reconstruction_results.png'.
"""

import os
import urllib.request
import gzip
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# =====================================================================
# PHASE 1: DATASET DOWNLOADING AND PARSING (PURE NUMPY)
# =====================================================================
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
DATA_DIR = "./data_raw"

def download_file(filename):
    """Downloads a file from the MNIST mirror if it doesn't already exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        url = MNIST_MIRROR + filename
        print(f"[*] Downloading {filename} from {url}...")
        urllib.request.urlretrieve(url, filepath)
    return filepath

def load_mnist_images(filepath):
    """Parses raw idx3-ubyte MNIST image files into a normalized NumPy array."""
    print(f"[*] Parsing {os.path.basename(filepath)}...")
    with gzip.open(filepath, 'rb') as f:
        # Read the IDX file header (magic number, number of images, height, width)
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.dtype('>i4'))
        # Read the raw byte data and normalize pixel values to [0.0, 1.0]
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.astype(np.float32) / 255.0
        # Reshape to (number of images, height * width)
        return images.reshape(num_images, rows * cols)

# Downloader routine (requires BypassSandbox: true if run)
try:
    print("[*] Retrieving MNIST images...")
    train_img_path = download_file("train-images-idx3-ubyte.gz")
    test_img_path = download_file("t10k-images-idx3-ubyte.gz")
    
    X_train = load_mnist_images(train_img_path)
    X_test = load_mnist_images(test_img_path)
    print(f"[*] Dataset successfully loaded into memory!")
    print(f"    - Training set shape: {X_train.shape}")
    print(f"    - Testing set shape:  {X_test.shape}\n")
except Exception as e:
    print(f"[!] Error downloading or parsing MNIST dataset: {e}")
    print("[!] Ensure you are connected to the internet. Retrying or raising.")
    raise e


# =====================================================================
# PHASE 2: MODEL ARCHITECTURE (DEEP AUTOENCODER IN NUMPY)
# =====================================================================
class NumPyAutoencoder:
    """
    A 3-Layer Deep Autoencoder in pure NumPy.
    Architecture:
      - Encoder: Linear(784 -> 64) -> ReLU -> Linear(64 -> 32) -> ReLU (Latent Bottleneck)
      - Decoder: Linear(32 -> 64) -> ReLU -> Linear(64 -> 784) -> Sigmoid (Reconstructed Output)
    """
    def __init__(self, input_dim=784, hidden_dim=64, latent_dim=32):
        # He (Kaiming) Normal Initialization for weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        
        self.W2 = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, latent_dim))
        
        self.W3 = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / latent_dim)
        self.b3 = np.zeros((1, hidden_dim))
        
        self.W4 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / hidden_dim)
        self.b4 = np.zeros((1, input_dim))
        
        # Adam Optimizer parameters
        self.m = {key: 0.0 for key in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4']}
        self.v = {key: 0.0 for key in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4']}
        self.t = 0
        
    def relu(self, x):
        return np.maximum(0, x)
        
    def sigmoid(self, x):
        # Stable sigmoid implementation to avoid overflow
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        """Forward pass through the Encoder and Decoder networks."""
        # --- ENCODER ---
        self.a0 = X  # Input shape: (Batch Size, 784)
        
        self.z1 = np.dot(self.a0, self.W1) + self.b1  # (Batch Size, 64)
        self.a1 = self.relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # (Batch Size, 32)
        self.a2 = self.relu(self.z2)  # Latent compressed representation
        
        # --- DECODER ---
        self.z3 = np.dot(self.a2, self.W3) + self.b3  # (Batch Size, 64)
        self.a3 = self.relu(self.z3)
        
        self.z4 = np.dot(self.a3, self.W4) + self.b4  # (Batch Size, 784)
        self.a4 = self.sigmoid(self.z4)  # Reconstructed output
        
        return self.a2, self.a4

    def backward(self, X):
        """Backward pass: Manual calculation of gradients using chain rule."""
        m_batch = X.shape[0]
        
        # Output Loss: MSE derivative
        # Loss = (1/784) * mean((a4 - X)^2)
        # d_Loss / d_a4 = (2 / (m_batch * 784)) * (a4 - X)
        da4 = (2.0 / (m_batch * 784)) * (self.a4 - X)
        
        # Output Activation: Sigmoid derivative
        # d_a4 / d_z4 = a4 * (1 - a4)
        dz4 = da4 * self.a4 * (1.0 - self.a4)
        
        # Layer 4 gradients
        dW4 = np.dot(self.a3.T, dz4)
        db4 = np.sum(dz4, axis=0, keepdims=True)
        da3 = np.dot(dz4, self.W4.T)
        
        # Layer 3 gradients (ReLU activation)
        dz3 = da3 * (self.z3 > 0)
        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)
        da2 = np.dot(dz3, self.W3.T)
        
        # Layer 2 gradients (ReLU activation)
        dz2 = da2 * (self.z2 > 0)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(dz2, self.W2.T)
        
        # Layer 1 gradients (ReLU activation)
        dz1 = da1 * (self.z1 > 0)
        dW1 = np.dot(self.a0.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        return {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3,
            'W4': dW4, 'b4': db4
        }

    def update_parameters(self, grads, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """Updates weights using the Adam Optimization algorithm in pure NumPy."""
        self.t += 1
        keys = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4']
        
        for key in keys:
            # 1. Update biased first moment estimate
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * grads[key]
            # 2. Update biased second raw moment estimate
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (grads[key] ** 2)
            
            # 3. Compute bias-corrected first moment estimate
            m_corrected = self.m[key] / (1 - beta1 ** self.t)
            # 4. Compute bias-corrected second raw moment estimate
            v_corrected = self.v[key] / (1 - beta2 ** self.t)
            
            # 5. Apply updates to the weights/biases
            param = getattr(self, key)
            new_param = param - lr * m_corrected / (np.sqrt(v_corrected) + eps)
            setattr(self, key, new_param)


# =====================================================================
# PHASE 3: TRAINING LOOP (NUMPY)
# =====================================================================
# Instantiate NumPy Model
model = NumPyAutoencoder(input_dim=784, hidden_dim=64, latent_dim=32)

EPOCHS = 10
BATCH_SIZE = 128
num_samples = X_train.shape[0]
num_batches = num_samples // BATCH_SIZE

print(f"[*] Starting NumPy Autoencoder training for {EPOCHS} epochs...")

for epoch in range(1, EPOCHS + 1):
    # Shuffle dataset indices at the start of each epoch
    shuffled_indices = np.random.permutation(num_samples)
    epoch_loss = 0.0
    
    for b in range(num_batches):
        # Extract the current batch
        batch_indices = shuffled_indices[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        X_batch = X_train[batch_indices]
        
        # 1. Forward Pass
        latent_vectors, reconstructions = model.forward(X_batch)
        
        # Calculate batch MSE Loss
        loss = np.mean((reconstructions - X_batch) ** 2)
        epoch_loss += loss * BATCH_SIZE
        
        # 2. Backward Pass
        grads = model.backward(X_batch)
        
        # 3. Update weights using Adam
        model.update_parameters(grads, lr=0.001)
        
    # Calculate average losses
    avg_train_loss = epoch_loss / (num_batches * BATCH_SIZE)
    
    # Calculate validation loss on unseen test set
    _, test_reconstructions = model.forward(X_test)
    avg_test_loss = np.mean((test_reconstructions - X_test) ** 2)
    
    print(f"    Epoch [{epoch}/{EPOCHS}] | Train MSE: {avg_train_loss:.6f} | Test MSE: {avg_test_loss:.6f}")

print("[*] Training completed successfully!\n")


# =====================================================================
# PHASE 4: VISUALIZATION & OUTPUT
# =====================================================================
print("[*] Generating original vs. reconstructed digit visualizations...")
# Use a fresh, unseeded generator so that different images are plotted EVERY time you run the script,
# while keeping the neural network weights and training determinism intact!
rng = np.random.default_rng()
random_indices = rng.choice(len(X_test), size=10, replace=False)

# Get reconstructed images for the test set
_, X_test_reconstructed = model.forward(X_test)

# Plot 10 sample digits
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(14, 4))
fig.suptitle("NumPy Fallback Autoencoder Image Reconstruction\nTop Row: Original MNIST Images | Bottom Row: Reconstructed (Compressed to 32 Dimensions)", fontsize=14, fontweight='bold')

for i, idx in enumerate(random_indices):
    # Plot Original Images
    axes[0, i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
    axes[0, i].get_xaxis().set_visible(False)
    axes[0, i].get_yaxis().set_visible(False)
    if i == 0:
        axes[0, i].set_ylabel("Original", fontsize=12, fontweight='bold')
    
    # Plot Reconstructed Images
    axes[1, i].imshow(X_test_reconstructed[idx].reshape(28, 28), cmap='gray')
    axes[1, i].get_xaxis().set_visible(False)
    axes[1, i].get_yaxis().set_visible(False)
    if i == 0:
        axes[1, i].set_ylabel("Reconstructed", fontsize=12, fontweight='bold')

plt.tight_layout()

# Save the visualization plot
output_filename = "reconstruction_results.png"
plt.savefig(output_filename, dpi=300)
print(f"[*] Visual comparisons saved successfully as: {os.path.abspath(output_filename)}")

# Summary stats for academic presentation
print("\n" + "="*60)
print("             SUMMARY OF COMPRESSION STATS (NUMPY FALLBACK)")
print("="*60)
print(f"1. Input Resolution:      28 x 28 pixels = 784 dimensions (grayscale)")
print(f"2. Latent Bottleneck:      32 floats")
print(f"3. Spatial Compression:    24.5x reduction")
print(f"4. Model MSE Loss:         {avg_test_loss:.5f} (on unseen test set)")
print(f"5. Saved Plot:             {output_filename}")
print("="*60 + "\n")
