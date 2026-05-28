# Deep Learning Image Compression: CNN Autoencoder Exploration

Welcome to your academic and research-oriented exploration of **Deep Learning-based Image Compression and Reconstruction**! This project implements a Convolutional Autoencoder (CNN Autoencoder) in PyTorch to compress images from the handwritten digit dataset (MNIST) down to a compact bottleneck representation and reconstruct them with minimal loss.

This repository serves as a foundational step for researchers and students exploring deep-learning-based image/video coding systems (such as those replacing JPEG, WebP, or standard H.264/HEVC/VVC components).

---

## Table of Contents
1. [Core Deep Learning Concepts Explained](#1-core-deep-learning-concepts-explained)
2. [Why CNNs Instead of Normal Dense (FC) Layers?](#2-why-cnns-instead-of-normal-dense-fc-layers)
3. [Autoencoder Architecture Details](#3-autoencoder-architecture-details)
4. [Step-by-Step Component Walkthrough](#4-step-by-step-component-walkthrough)
5. [How to Run the Code](#5-how-to-run-the-code)
6. [Expected Outputs & Research Metrics](#6-expected-outputs--research-metrics)
7. [Academic Presentation Guide (Slide-by-Slide)](#7-academic-presentation-guide-slide-by-slide)

---

## 1. Core Deep Learning Concepts Explained

### Encoder-Decoder Architecture
In neural-network-based compression, the system is split into two complementary halves:
* **The Encoder ($E$):** Maps the high-dimensional input image $x \in \mathbb{R}^{D}$ to a low-dimensional compressed feature vector $z \in \mathbb{R}^{d}$ (where $d \ll D$). Formally: $z = E(x)$.
* **The Decoder ($D$):** Maps the low-dimensional latent vector $z$ back to the original high-dimensional space, reconstructing the output image $\hat{x} \in \mathbb{R}^{D}$. Formally: $\hat{x} = D(z) = D(E(x))$.

### Latent Space (The Bottleneck)
The **Latent Space** is the low-dimensional vector space $\mathbb{R}^{d}$ where the compressed representation of the input data lives. It is called "latent" (hidden) because the network must *learn* an optimal coordinate system to represent the most vital characteristics of the image (e.g., shape, line orientations, stroke width) rather than storing individual, redundant pixels.

### Compression Ratio (CR)
The **Spatial Compression Ratio (CR)** quantifies how much we have shrunk the spatial representations. It is defined as:
$$\text{Compression Ratio (CR)} = \frac{\text{Dimensionality of Input Image } (D)}{\text{Dimensionality of Latent Space } (d)}$$

For our specific MNIST implementation:
* Input Image dimensions: $1 \times 28 \times 28 = 784$ floating-point values.
* Latent vector dimension: $32$ floating-point values.
$$\text{CR} = \frac{784}{32} = 24.5$$
This represents a **24.5x reduction** in spatial features. The autoencoder discards **95.92%** of the input dimensions, keeping only the most important 4.08%!

### Reconstruction Quality (MSE vs. PSNR)
* **Mean Squared Error (MSE):** The standard training loss. It measures the average squared difference between original pixels $x_i$ and reconstructed pixels $\hat{x}_i$:
$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N} (x_i - \hat{x}_i)^2$$
* **Peak Signal-to-Noise Ratio (PSNR):** The standard metric in compression research (measured in decibels, dB). It is inversely proportional to logarithmic MSE:
$$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right)$$
*(For pixel values normalized between 0 and 1, $\text{MAX}_I = 1.0$)*. In academic papers, a higher PSNR (typically $>30$ dB) signifies better reconstruction quality.

---

## 2. Why CNNs Instead of Normal Dense (FC) Layers?

Traditional Multi-Layer Perceptrons (MLPs) use **Fully Connected (Dense) layers** where every neuron in layer $L$ connects to every neuron in layer $L-1$. For image processing, this is mathematically and computationally inefficient for three key reasons:

```
  Dense (Fully Connected) Layer            Convolutional Layer (Local & Shared)
       [Pixel 1] \   / [Neuron 1]              [Pixel 1-9] -- (Kernel) -- [Feature A]
       [Pixel 2] - X - [Neuron 2]              [Pixel 10-18] - (Kernel) - [Feature B]
       [Pixel 3] /   \ [Neuron 3]              * Shared weights scanned across image!
   * Parameter explosion; ignores shape        * Preserves 2D geometry and local details
```

### A. Local Spatial Receptive Fields
An image has highly localized structures: a pixel is highly related to its immediate neighboring pixels, but has little relation to pixels on the opposite side of the image.
* **Dense Layers:** Treat an image as a flat 1D vector, destroying the 2D spatial arrangement.
* **CNNs:** Use a local kernel (e.g., $3 \times 3$ or $5 \times 5$) that only processes nearby pixels, preserving regional spatial context.

### B. Parameter Sharing & Size Reduction
Let's compare the parameters required to extract features from a single grayscale $28 \times 28$ image:
1. **Fully Connected approach:** To project 784 inputs to 784 outputs requires:
$$\text{Weights} = 784 \times 784 = 614,656 \text{ parameters}$$
2. **Convolutional approach (Conv2d with 16 filters of size $3 \times 3$):**
$$\text{Weights} = \text{filters} \times (\text{input\_channels} \times \text{kernel\_height} \times \text{kernel\_width}) = 16 \times (1 \times 3 \times 3) = 144 \text{ parameters}$$
CNNs drastically reduce parameter counts, preventing overfitting and making deep networks trainable.

### C. Translation Invariance
If a handwritten digit "3" is shifted 3 pixels to the right, a Fully Connected network sees completely new input values and may fail. A CNN scans the exact same filter kernel across the entire image, detecting the visual characteristics of "3" regardless of where it appears.

---

## 3. Autoencoder Architecture Details

```
Input Image (1x28x28)
     │
     ▼  [Encoder]
Conv2d (1 -> 16, stride=2, padding=1)  --> Output: 16x14x14
     │
     ▼  [ReLU]
Conv2d (16 -> 32, stride=2, padding=1) --> Output: 32x7x7
     │
     ▼  [ReLU + Flatten]
Linear (1568 -> 32)
     │
     ▼
 Latent Space Bottleneck (32 dimensions) <-- 24.5x Spatial Compression
     │
     ▼  [Decoder]
Linear (32 -> 1568)
     │
     ▼  [ReLU + Unflatten]
ConvTranspose2d (32 -> 16, stride=2, padding=1) --> Output: 16x14x14
     │
     ▼  [ReLU]
ConvTranspose2d (16 -> 1, stride=2, padding=1)  --> Output: 1x28x28
     │
     ▼  [Sigmoid]
Reconstructed Output (1x28x28)
```

---

## 4. Step-by-Step Component Walkthrough

### A. Dataset Loading
We use `torchvision.datasets.MNIST`. The images are transformed to tensors using `transforms.ToTensor()`, which scales the pixel values from the raw integers $[0, 255]$ to floating-point values in $[0.0, 1.0]$. The `DataLoader` batches the dataset into groups of 128 images, which are shuffled every epoch during training to ensure uniform gradient steps.

### B. CNN Encoder
The encoder applies two convolutional layers with a stride of 2. Stride 2 functions as a learned downsampling operation, reducing spatial width and height by 50% per layer ($28 \to 14 \to 7$). A final `Linear` layer compresses the resulting 1,568-dimensional flattened features into 32 values.

### C. Latent Space (The Bottleneck)
The bottleneck is a 32-element vector. This is the **fully compressed file** in deep-learning-based codecs.

### D. CNN Decoder
The decoder's job is to reverse the encoder. It projects the 32-dimensional bottleneck back to 1,568 dimensions, unflattens it to $(32, 7, 7)$, and then uses two `ConvTranspose2d` (fractionally strided convolution) layers. These layers expand spatial dimensions ($7 \to 14 \to 28$) using learnable filters.

### E. Reconstruction & Loss Function
The final decoder layer uses a `Sigmoid` activation function, clamping values between $0.0$ and $1.0$ (perfectly representing gray levels).
We use **Mean Squared Error (MSE) Loss**, comparing original pixel values $x$ with output reconstructions $\hat{x}$.

### F. Optimizer & Training Loop
We use the **Adam Optimizer** (learning rate = $0.001$).
In each training step:
1. `optimizer.zero_grad()` clears historical gradient calculations.
2. `model(images)` performs the forward pass.
3. `criterion(reconstructions, images)` calculates the reconstruction MSE.
4. `loss.backward()` propagates gradients backward through the network (Backpropagation).
5. `optimizer.step()` updates model weights.

---

## 5. How to Run the Code

### Prerequisites
Make sure you have PyTorch, Torchvision, and Matplotlib installed in your Python environment:
```bash
pip install torch torchvision matplotlib numpy
```

### Running the Training Script
Run the self-contained script using Python:
```bash
python train.py
```

### Outputs Generated
* **Terminal Logs:** Showing device selection, detailed tensor shapes, compression calculations, and loss per epoch.
* **Visualization File:** `reconstruction_results.png` showing the side-by-side reconstruction comparison.

---

## 6. Expected Outputs & Research Metrics

### Training Progress Logs
As the training progresses over 5 epochs, the loss (MSE) on both the training set and unseen test set will steadily decrease:
* **Epoch 1:** MSE $\approx 0.05$ to $0.07$ (Images are blurry, rough shapes appear).
* **Epoch 3:** MSE $\approx 0.02$ to $0.03$ (Images become legible, background noise reduces).
* **Epoch 5:** MSE $\approx 0.01$ or lower (Reconstructed images are highly legible, showing sharp strokes).

### Visualization Results (`reconstruction_results.png`)
The saved image showcases a direct visual mapping:
* **Top Row (Originals):** Clean, crisp handwritten digits from the test set.
* **Bottom Row (Reconstructed):** Decoded images reconstructed from the 32-dimensional bottleneck. They match the shape and orientation of the originals, showing that the network successfully learned to compress handwritten geometry!

---

## 7. Academic Presentation Guide (Slide-by-Slide)

If you are presenting this implementation for an academic class or research project, use this highly structured slide outline to frame your presentation:

### Slide 1: Title Slide
* **Title:** Deep Learning-based Image Coding: Exploring Convolutional Autoencoders
* **Subtitle:** An Empirical Study of Reconstruction Quality on Handwritten Digits
* **Presenter:** [Your Name]
* **Key Visuals:** A simple diagram showing an input image shrinking into a barcode-like latent vector, then expanding.

### Slide 2: The Core Problem: Traditional vs. Learned Compression
* **Bullet Points:**
  - **Traditional Codecs (JPEG, HEVC, VVC):** Rely on human-engineered algorithms, block-based discrete cosine transforms (DCT), and hand-crafted quantization tables.
  - **Learned Codecs (Autoencoders):** Replace hand-crafted components with artificial neural networks that learn optimal, non-linear representations directly from raw training data.
  - **Research Importance:** The Next Generation of Image/Video Compression standards (e.g., MPEG-JVET) relies heavily on neural architectures for high compression gains.

### Slide 3: Concept Definitions & Terminology
* **Bullet Points:**
  - **Encoder-Decoder:** $z = E(x)$ (compression) and $\hat{x} = D(z)$ (reconstruction).
  - **Latent Space Bottleneck:** A compact 32-dimensional vector capturing key semantic features.
  - **Spatial Compression Ratio (CR):** 784 spatial elements to 32 latent elements = $24.5\times$ spatial feature reduction.
  - **Objective Evaluation:** Mean Squared Error (MSE) representing reconstruction fidelity.

### Slide 4: Why Convolutions? (Mathematical Justification)
* **Bullet Points:**
  - Fully Connected layers fail on images due to **parameter explosion** ($28 \times 28 \to 28 \times 28$ requires $>614,000$ weights).
  - Convolutions use **Local Receptive Fields** to respect 2D spatial relationships.
  - Convolutions exploit **Weight Sharing** to reduce parameters down to just $144$ weights per layer, ensuring highly efficient extraction of shapes, edges, and strokes.
  - **Translation Invariance:** Moving a digit inside the frame does not affect the model's ability to compress and reconstruct it.

### Slide 5: Proposed Network Architecture
* **Bullet Points:**
  - Symmetric CNN Encoder-Decoder structure in PyTorch.
  - **Downsampling:** Achieved in the Encoder via strided convolutions (`stride=2`), mapping $28 \times 28 \to 14 \times 14 \to 7 \times 7$.
  - **Upsampling:** Rebuilt in the Decoder using learned fractional transposed convolutions (`ConvTranspose2d`), mapping $7 \times 7 \to 14 \times 14 \to 28 \times 28$.
  - **Activation Bounds:** `ReLU` for hidden representation layers; `Sigmoid` mapping output intensities safely to $[0, 1]$.

### Slide 6: Training Methodology & Parameters
* **Bullet Points:**
  - **Dataset:** MNIST Grayscale Images (60,000 train, 10,000 test).
  - **Loss Criterion:** Pixel-wise Mean Squared Error (MSE).
  - **Optimizer:** Adam (Adaptive learning rates, $\eta = 0.001$).
  - **Batch Size:** 128 samples per optimization step.
  - **Training Duration:** 5 epochs (to demonstrate fast convergence).

### Slide 7: Results and Visual Analysis (Showcase Slide!)
* **Key Visuals:** Insert your generated `reconstruction_results.png`.
* **Bullet Points:**
  - Highlight the progressive descent of MSE loss (e.g., Test MSE reaching $\approx 0.01$).
  - Discuss the clarity of reconstructed numbers. 
  - Point out that even with a **24.5x feature reduction**, the reconstructed numbers retain their shape, character identity, and specific stroke details.

### Slide 8: Future Directions & Scope Limitations
* **Bullet Points:**
  - **Current Scope:** Baseline lossy image compression and spatial dimension reduction using MNIST.
  - **Research Next Steps:**
    1. **Varying Latent Dimensions:** Analyzing the rate-distortion curve (how increasing/decreasing latent size affects reconstruction quality).
    2. **Quantization & Entropy Coding:** Converting continuous latent float values into integers to measure actual compressed file sizes in Kilobytes (using arithmetic coding).
    3. **More Complex Datasets:** Scaling up to color images (CIFAR-10, Kodak Dataset) using deeper networks and residual structures.

### Slide 9: Conclusion & Q&A
* **Bullet Points:**
  - Convolutional Autoencoders represent a highly efficient paradigm for neural image compression.
  - PyTorch allows for clean, modular, and performant encoder-decoder implementations.
  - Thank you! Questions are welcome.
