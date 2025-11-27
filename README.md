# DCNet: Dark Channel Network for Single-Image Dehazing
<img width="911" height="436" alt="image" src="https://github.com/user-attachments/assets/2cd9e9c8-1111-4a9f-b437-35d4d3c6b6a6" />
## Overview
**DCNet (Dark Channel Network)** is a deep learning-based architecture designed for **single-image dehazing**.  
It leverages the **dark channel prior** along with the **value (V) channel** from HSV color space to estimate the **transmission map**, which is then used to restore a clear haze-free image using the **atmospheric scattering model**.

This model was originally proposed in:
> Akshay Bhola, Teena Sharma, Nishchal K. Verma.  
> *DCNet: Dark Channel Network for Single-Image Dehazing*.  
> Published in *Springer Nature, 2021*.

---

## Problem Statement
Outdoor images captured in hazy conditions suffer from:
- Reduced contrast and visibility
- Color distortion
- Poor performance in computer vision tasks like detection & segmentation  

**Single-image dehazing** is an *ill-posed problem* because:
- The **clear scene radiance**
- The **atmospheric light**
- The **transmission map**  
are all unknown and must be estimated from just one hazy image.

---

##Architeture of DC Net:-
<img width="930" height="291" alt="image" src="https://github.com/user-attachments/assets/8f19e050-34ed-4aba-8f53-d8680cb2124b" />

DCNet consists of **two major components**:

### 1. Feature Extraction Layer
Extracts haze-relevant features from the input image:
- **Value (V) channel**: Captures brightness and edge/textural information.
- **Multi-scale Dark Channels**: Computed over multiple patch sizes  
  *(1×1, 3×3, 5×5, 7×7, 10×10)*.  
  - Small patches → good edge information but weaker haze assumption.  
  - Large patches → better haze density capture but may cause halo effects.  
  - **Multi-scale fusion** balances both.

The extracted features are concatenated to form the input for CNN.

### 2. Convolutional Neural Network Layer
A lightweight **5-layer CNN** estimates the **transmission map**.  

| Layer | Filters | Kernel Size |
|-------|---------|-------------|
| Conv1 | 64      | 3×3 |
| Conv2 | 32      | 3×3 |
| Conv3 | 16      | 3×3 |
| Conv4 | 8       | 3×3 |
| Conv5 | 1       | 3×3 |

- Activation: **Leaky ReLU (slope=0.05)**  
- Padding: Zero-padding to preserve dimensions  
- Output: Transmission map `t(x, y)`

---

## Dehazing Process
The estimated transmission map `t(x, y)` is used in the **atmospheric scattering model**:
J(x, y) = ( S(x, y) - A * (1 - t(x, y)) ) / t(x, y)

where:
- `J(x, y)` = Dehazed image (clear scene radiance)  
- `S(x, y)` = Input hazy image  
- `A` = Global atmospheric light (kept constant as `(0.95, 0.95, 0.95)`)  

---

## Training

- **Data:** 200 images from RESIDE SOTS dataset.
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** RMSprop
- **Variants:**
  - Model A: Trained with transmission `t ∈ [0.4, 1.0]` → Robust for heavy haze
  - Model B: Trained with `t ∈ [0.7, 1.0]` → Better for light haze

---

## Results

### Quantitative Metrics
Evaluated using:
- **MSE (Mean Squared Error)**
- **SSIM (Structural Similarity Index)**
- **PSNR (Peak Signal-to-Noise Ratio)**


## Hybrid Physics-Inspired ViT for Image Dehazing

This project introduces a hybrid Vision Transformer (ViT) + physics-driven dehazing architecture that significantly outperforms classical CNN-based models like DCNet, AOD-Net, DehazeNet, and MSCNN.
- The model fuses:
- Physics-based priors (Value channel + multi-scale Dark Channel Prior)
- Global attention from a pretrained ViT-Base
- MAE-style decoder for high-quality reconstruction
- A learnable 7→3 projection layer to inject physics priors into ViT

## Architecure
-  Full DCNet-style Physics Module
- Extracted 7 haze-aware channels per image:
- 1× Value (V) channel
- 6× Multi-scale Dark Channel Priors (3×3 → 13×13)
- Produces a 7-channel physics feature map.
- Hybrid ViT-Based Dehazing Architecture
- 7→3 Projection Layer
- Learnable Conv2D
- Converts 7 physics channels → 3 pseudo-RGB
- Makes the model HuggingFace ViT compatible

## Vision Transformer Encoder

- ViT-Base Patch-16
- 90% frozen
- Only last 2 Transformer blocks trainable
- Prevents overfitting on small datasets

## MAE Decoder

- Patch-level reconstruction
- Learns global + local dehazing patterns


## Architecture Diagram

Hazy Image
     ↓
RGB → Resize → Tensor
     ↓
7-Channel Physics Feature Extraction (V + 6×DCP)
     ↓
Learnable 7→3 Conv Projection
     ↓
Vision Transformer Encoder (Frozen except last 2 layers)
     ↓
MAE Decoder
     ↓
Reconstructed Clean Image

## Training Details
- ViT Encoder	Frozen (except last 2 blocks)
- Projection Layer (7→3)	Trainable
- MAE Decoder	Trainable
- Epochs	15–25
- Loss	L1 Loss
- Optimizer	Adam (lr = 2e-4)
- Dataset	1,000 images (RESIDE-6K)
- Patch Size	16×16






