# Brain Tumor Detection via Deep Learning with Topological Data Analysis

## Introduction

Brain tumor detection from MRI imaging is a critical task in medical diagnosis, where accurate and interpretable classification can significantly impact patient outcomes. Traditional convolutional neural networks (CNNs) have demonstrated strong performance in medical image classification tasks [1, 2], but they often lack interpretability and may miss subtle structural patterns that are clinically relevant.

This work presents a novel two-stage architecture that combines deep learning with Topological Data Analysis (TDA) for brain tumor detection. Our approach leverages VGG19 as a feature extractor, incorporates Gradient-weighted Class Activation Mapping (Grad-CAM++) [3] for attention-based feature localization, and applies persistent homology [4, 5] to capture topological features from attention maps. By fusing CNN-based features with topological descriptors, our model aims to achieve both high classification accuracy and enhanced interpretability.

The integration of TDA with deep learning has shown promise in various medical imaging applications [6, 7], as topological features can capture shape-based patterns that are invariant to certain transformations. Our architecture explicitly models these topological characteristics through Vietoris-Rips complexes and persistence diagrams, providing complementary information to standard CNN features.

### References
1. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems, 25.
3. Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018). Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks. 2018 IEEE Winter Conference on Applications of Computer Vision (WACV).
4. Edelsbrunner, H., & Harer, J. (2010). Computational topology: an introduction. American Mathematical Society.
5. Carlsson, G. (2009). Topology and data. Bulletin of the American Mathematical Society, 46(2), 255-308.
6. Qaiser, T., et al. (2019). Persistent homology for fast tumor segmentation in whole slide histology images. Procedia Computer Science, 90, 119-124.
7. Hofer, C., Kwitt, R., Niethammer, M., & Uhl, A. (2017). Deep learning with topological signatures. Advances in Neural Information Processing Systems, 30.

---

## Data

### Dataset Description

We utilize the Brain MRI Images for Brain Tumor Detection dataset, publicly available on Kaggle:

**Dataset URL:** [https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

The dataset contains MRI scans of the brain classified into two categories:
- **Tumor (Positive):** MRI images showing presence of brain tumors
- **No Tumor (Negative):** MRI images without tumors

### Data Preprocessing

All images are preprocessed using the following pipeline:

1. **Resizing:** Images are resized to 224×224 pixels to match VGG19 input requirements
2. **Normalization:** Pixel values are scaled to the range [0, 1]
3. **Channel Format:** Images are converted to RGB format (3 channels)
4. **Data Augmentation:** Training images undergo extensive augmentation including:
   - Random horizontal and vertical flips
   - Random rotation (±15 degrees)
   - Random affine transformations (translation, scaling, shear)
   - Color jittering (brightness, contrast, saturation, hue)
   - Gaussian blur
   - Random erasing

### Data Split

The dataset is partitioned using stratified sampling to maintain class balance:
- **Training Set:** 60% of the data
- **Validation Set:** 20% of the data
- **Test Set:** 20% of the data

Data augmentation is applied with a multiplier of 2×, effectively tripling the training set size (original + 2 augmented versions per image).

---

## Methods

### Architecture Overview

Our proposed architecture consists of three main components operating in a two-stage training paradigm: (1) a CNN branch based on VGG19 for hierarchical feature extraction, (2) a topology branch that extracts structural features via TDA from Grad-CAM++ attention maps, and (3) a fusion branch that combines both feature representations for final classification.

![Main Architecture](./figures/architecture_main.pdf)
*Figure 1: Overall architecture showing VGG19 backbone, CNN branch, topology branch (with GradCAM++ and TDA), and fusion branch.*

### Stage 1: CNN Branch

#### VGG19 Backbone

We employ VGG19 [1] pretrained on ImageNet as our feature extraction backbone. VGG19 is selected for its strong performance on medical imaging tasks and its interpretability due to its sequential convolutional structure. The first 15 layers are frozen to preserve low-level feature representations learned from natural images, while the remaining layers are fine-tuned on our brain MRI dataset. This transfer learning approach has been shown to be effective for medical imaging tasks with limited training data [8].

The VGG19 features layer outputs a feature map of size 512×7×7, which is then passed through a dropout layer (p=0.5) for regularization.

#### Fully Connected Layers

The CNN branch consists of two fully connected layers:

1. **FC-512:** Linear layer (25088 → 512) followed by batch normalization and ReLU activation
2. **FC-128:** Linear layer (512 → 128) followed by ReLU activation

Batch normalization [9] is applied to stabilize training and improve convergence. The progressive dimensionality reduction (25088 → 512 → 128) compresses the high-dimensional feature space while retaining discriminative information. 

Note that dropout is not applied in the CNN branch, as regularization is primarily achieved through weight decay (L2), batch normalization, and aggressive data augmentation. During Stage 1 training, this branch is connected to a sigmoid output layer for binary classification.

### Stage 2: Topology Branch with Grad-CAM++

#### Motivation for Topological Features

While CNNs excel at learning hierarchical features, they may not explicitly capture topological properties such as connected components, holes, and voids that are medically significant in tumor characterization. Persistent homology [4, 5] provides a mathematically rigorous framework for quantifying such topological structures across multiple scales. By integrating TDA with deep learning, we aim to enhance the model's ability to recognize shape-based patterns that complement texture and intensity features learned by the CNN.

#### Grad-CAM++ for Attention Extraction

Grad-CAM++ [3] is employed to generate class-discriminative attention maps from the VGG19 backbone. Unlike standard Grad-CAM, Grad-CAM++ provides better localization for multiple objects and produces more visually coherent attention maps through pixel-wise weighting of gradients. The attention map highlights regions of the input image that most strongly influence the classification decision, effectively serving as a learned saliency detector.

Formally, given the final convolutional feature maps $A^k$ and the gradients of the class score $y^c$ with respect to $A^k$, Grad-CAM++ computes weighted combinations using:

```
$α^kc = Σ_i Σ_j (∂²y^c / ∂A^k_ij²) / (2(∂²y^c / ∂A^k_ij²) + Σ_i Σ_j A^k_ij (∂³y^c / ∂A^k_ij³))$
```

The resulting attention map is upsampled to the input resolution (224×224) and normalized to [0, 1].

**Important implementation detail:** During Stage 2 training, Grad-CAM++ is computed on detached inputs, and the resulting attention maps are also detached before being fed to the topology branch. This design choice ensures stable training by isolating the topology branch optimization from the CNN feature extraction, preventing gradient backpropagation through the attention mechanism. This gradient isolation allows each component to specialize independently without destabilizing the learned CNN features.

#### Point Cloud Sampling

From the Grad-CAM++ attention map, we extract a point cloud representation by sampling the top-k highest attention regions. Specifically, we select 400 points corresponding to the pixels with the highest attention weights. Each point is represented by its normalized 2D coordinates (x, y) ∈ [0, 1]². This point cloud serves as the input to the topological analysis pipeline.

The choice of 400 points balances computational efficiency with sufficient topological resolution to capture meaningful structures. This sampling strategy has been successfully applied in prior work combining deep learning with TDA [7].

#### Vietoris-Rips Complex and Persistent Homology

We construct a Vietoris-Rips (VR) complex [4] on the sampled point cloud. The VR complex is a simplicial complex built by connecting points whose pairwise distances are below a threshold parameter ε. As ε increases, the complex grows, and topological features (connected components, loops, voids) appear and disappear. Persistent homology tracks these features across the filtration, summarizing their "lifetime" in persistence diagrams.

We compute homology groups H₀ (connected components) and H₁ (loops/holes), which are particularly relevant for tumor detection:
- **H₀ features** capture the number and distribution of attention clusters
- **H₁ features** capture ring-like structures and holes that may correspond to tumor boundaries or necrotic regions

The use of persistent homology has been shown to provide stable and discriminative features for medical image analysis [6, 10].

#### Persistence Images

Persistence diagrams are vectorized using persistence images [11], a stable and differentiable representation suitable for machine learning. Each persistence diagram is converted into a 64×64 grid by placing Gaussian kernels at the birth-death coordinates of each topological feature, weighted by their persistence (lifetime). This process yields two 64×64 images for H₀ and H₁, which are flattened and concatenated to form an 8192-dimensional feature vector.

![Topology Compressor Details](./figures/topology_compressor_detail.pdf)
*Figure 2: Detailed architecture of the topology compressor showing the transformation from persistence images (8192-dim) through three fully connected layers with layer normalization and dropout to the final 128-dimensional topology features.*

#### Topology Compressor

The 8192-dimensional persistence image features are compressed through a three-layer network:

1. **Layer 1:** Linear (8192 → 512) + LayerNorm + ReLU + Dropout(0.3)
2. **Layer 2:** Linear (512 → 256) + LayerNorm + ReLU + Dropout(0.2)
3. **Layer 3:** Linear (256 → 128)

Layer normalization [12] is used instead of batch normalization to provide more stable training for the topology branch. Dropout is strategically applied only in the topology branch (not in CNN/fusion branches) because the high-dimensional persistence features (8192-dim) are particularly prone to overfitting. The dropout rates (0.3 and 0.2) are chosen to prevent overfitting while maintaining gradient flow. The final 128-dimensional output matches the dimensionality of the CNN branch features, facilitating balanced fusion.

### Stage 3: Fusion Branch

The fusion branch combines the 128-dimensional CNN features with the 128-dimensional topology features through concatenation, yielding a 256-dimensional joint representation. This combined feature vector is processed through two fully connected layers:

1. **Fusion FC-256:** Linear (256 → 256) + BatchNorm + ReLU
2. **Fusion FC-128:** Linear (256 → 128) + ReLU

Finally, a sigmoid-activated linear layer (128 → 1) produces the binary classification output.

Note that dropout is not applied in the fusion branch, as sufficient regularization is achieved through:
- Weight decay (L2 regularization)
- Batch normalization
- The two-stage training paradigm that prevents overfitting
- Gradient isolation in the topology branch

The fusion strategy allows the model to learn complementary representations: the CNN branch captures local texture and intensity patterns, while the topology branch captures global structural characteristics. This multi-view approach has been shown to improve robustness and generalization in medical imaging [13].

### Training Strategy

#### Two-Stage Training

We adopt a two-stage training approach to stabilize learning:

**Stage 1 (CNN-only):**
- Train only the CNN branch (VGG19 + FC layers) with frozen backbone layers
- Optimize with Adam optimizer (lr=1e-4, weight_decay=1e-4)
- Use ReduceLROnPlateau scheduler (factor=0.5, patience=7)
- Binary cross-entropy loss with early stopping (patience=20)
- Epochs: 50

**Stage 2 (Topology + Fusion):**
- Freeze CNN branch, train topology branch and fusion layers
- Grad-CAM++ computed in detached mode to prevent gradient backpropagation
- Optimize with Adam optimizer (lr=1e-4, weight_decay=1e-4)
- Use ReduceLROnPlateau scheduler (factor=0.5, patience=10)
- Binary cross-entropy loss with early stopping (patience=30)
- Epochs: 100

This staged approach prevents the topology branch from destabilizing the pre-trained CNN features and allows each component to specialize before integration.

#### Regularization

Multiple regularization techniques are employed to prevent overfitting:
- **Weight Decay (L2 regularization):** λ = 1e-4 on all trainable parameters
- **Dropout:** Applied strategically only in:
  - VGG19 backbone (p=0.5 after features layer)
  - Topology compressor (p=0.3 after first layer, p=0.2 after second layer)
- **Batch Normalization:** Applied in CNN branch (after FC-512) and fusion branch (after FC-256)
- **Layer Normalization:** Applied in topology compressor (after each linear layer)
- **Data Augmentation:** 2× multiplier on training data
- **Early Stopping:** Based on validation loss with patience thresholds
- **Two-Stage Training:** Prevents topology branch from destabilizing pre-trained CNN features
- **Gradient Isolation:** Detached Grad-CAM++ computation isolates topology optimization

The strategic placement of dropout reflects component-specific needs: the topology branch processes high-dimensional persistence features (8192-dim) prone to overfitting and thus requires explicit dropout, while CNN and fusion branches achieve sufficient regularization through batch normalization and weight decay given their more direct optimization paths.

### Implementation Details

The model is implemented in PyTorch using:
- VGG19 pretrained weights from torchvision
- torch-topological library for differentiable TDA operations
- Custom Grad-CAM++ implementation following [3]

Training is performed on GPU with batch size 8 and mixed precision (when available) to reduce memory consumption.

### References (Methods)
8. Tajbakhsh, N., et al. (2016). Convolutional neural networks for medical image analysis: Full training or fine tuning? IEEE Transactions on Medical Imaging, 35(5), 1299-1312.
9. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. International Conference on Machine Learning.
10. Clough, J. R., et al. (2020). A topological loss function for deep-learning based image segmentation using persistent homology. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(12), 8766-8778.
11. Adams, H., et al. (2017). Persistence images: A stable vector representation of persistent homology. Journal of Machine Learning Research, 18(8), 1-35.
12. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. arXiv preprint arXiv:1607.06450.
13. Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

---
