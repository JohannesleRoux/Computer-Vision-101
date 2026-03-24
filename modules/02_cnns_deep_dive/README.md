# Module 02 — CNNs Deep Dive

## Learning Objectives

1. Calculate the receptive field of a conv stack analytically.
2. Implement and compare depthwise separable, dilated, transposed, and group convolutions.
3. Understand ASPP (Atrous Spatial Pyramid Pooling) and its role in segmentation.
4. Build intuition for when each conv variant should be preferred.

---

## Key Concepts

### Receptive Field
The *effective receptive field* of a neuron is the region of the input image
that can influence its activation.  For a stack of 3×3 convolutions with stride 1,
the receptive field grows by 2 pixels per layer.  Dilated convolutions expand
the receptive field without increasing parameters.

Formula for a single layer: `RF_out = RF_in + (kernel_size - 1) * stride_product`

### Depthwise Separable Convolution (MobileNet)
Factorises a standard `k×k` convolution into:
1. **Depthwise** conv: `k×k` filter applied independently per channel.
2. **Pointwise** conv: `1×1` conv to combine channels.

Parameter savings: `k² × Cout / (k² + Cout)` vs. standard conv.

### Dilated (Atrous) Convolution
Inserts gaps (zeros) between kernel elements, controlled by dilation rate `r`.
Effective kernel size: `k_eff = k + (k-1)(r-1)`.
Receptive field grows without striding (preserves spatial resolution).

### Transposed Convolution (Deconvolution)
Learnable upsampling by `stride` factor.  Used in decoders (U-Net, FCN).
Note: can produce checkerboard artifacts — often replaced by bilinear upsample + conv.

### Group Convolution
Splits channels into `G` independent groups.  Standard conv is `G=1`,
depthwise conv is `G=C`.  Used in ResNeXt, ShuffleNet.

### ASPP — Atrous Spatial Pyramid Pooling
Applies parallel dilated convolutions with rates `{6, 12, 18}` + global pooling
to capture multi-scale context.  Central to DeepLab v3/v3+.

---

## Key References

- Howard et al. (2017) *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision*
- Yu & Koltun (2015) *Multi-Scale Context Aggregation by Dilated Convolutions*
- Zeiler et al. (2010) *Deconvolutional Networks*
- Xie et al. (2017) *Aggregated Residual Transformations for Deep Neural Networks* (ResNeXt)
- Chen et al. (2018) *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation* (DeepLab v3+)
