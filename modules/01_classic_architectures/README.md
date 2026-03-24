# Module 01 — Classic CNN Architectures

## Learning Objectives

1. Understand the historical progression from LeNet to ResNet and why each architecture was a breakthrough.
2. Implement LeNet-5, AlexNet, VGG, and ResNet blocks from scratch in PyTorch.
3. Recognise the vanishing-gradient problem and explain how residual connections solve it.
4. Apply transfer learning from a pretrained ResNet to a new classification task.

---

## Key Concepts

### LeNet-5 (LeCun et al., 1998)
The first practically successful CNN, designed for handwritten digit recognition (MNIST).
Architecture: Conv → Pool → Conv → Pool → FC → FC → Output.
Key insight: shared weights reduce parameter count vs. fully connected layers.

### AlexNet (Krizhevsky et al., 2012)
Won ImageNet LSVRC-2012 with a 10.8% top-5 error gap over the second place.
Innovations: ReLU activations, dropout regularisation, data augmentation, GPU training.

### VGG (Simonyan & Zisserman, 2014)
Demonstrated that depth is the key factor in performance.
Used only 3×3 convolutions stacked to build deep networks (VGG-16: 16 weight layers).
Two stacked 3×3 convs have the same receptive field as one 5×5 conv but fewer parameters.

### ResNet (He et al., 2015)
Introduced *residual (skip) connections*: `y = F(x) + x`.
Allows training of very deep networks (50, 101, 152 layers) by mitigating vanishing gradients.
Introduced batch normalisation between conv and ReLU.

### Transfer Learning
Pretrained features generalise across tasks.  Typical workflow:
1. Load a pretrained backbone (e.g., ResNet-50 on ImageNet).
2. Replace the final classification head for the new task.
3. Fine-tune either just the head (feature extraction) or all layers.

---

## Key References

- LeCun et al. (1998) *Gradient-Based Learning Applied to Document Recognition*. Proc. IEEE.
- Krizhevsky et al. (2012) *ImageNet Classification with Deep Convolutional Neural Networks*. NeurIPS.
- Simonyan & Zisserman (2014) *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR 2015.
- He et al. (2015) *Deep Residual Learning for Image Recognition*. CVPR 2016.
- [PyTorch torchvision model zoo](https://pytorch.org/vision/stable/models.html)
