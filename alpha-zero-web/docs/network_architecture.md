# Neural Network Architecture

The engine uses a dual-headed residual network (ResNet) inspired by AlphaZero, tailored for fast client-side inference in a web browser.

## Input Definition
* **Shape:** `[18, 8, 8]` (Channels-First: Planes × Height × Width)
* **Conversion:** The network instantly transposes the input to `[8, 8, 18]` (Channels-Last) internally via a `Permute` layer to maximize compatibility with TensorFlow/ONNX CPU execution backends.

## Network Topology
1. **Convolutional Block:**
   * Conv2D (Kernel: 3×3, Stride: 1, Padding: Same)
   * Batch Normalization
   * ReLU Activation
2. **Residual Blocks (Configurable, Default: 4):**
   * Conv2D (3×3) → BatchNorm → ReLU
   * Conv2D (3×3) → BatchNorm
   * Skip Connection (Add input) → ReLU
3. **Policy Head:**
   * Conv2D (Kernel: 1×1, Filters: 2) → BatchNorm → ReLU
   * Flatten
   * Dense (Size: 4672) → Softmax Activation (Action Probabilities)
4. **Value Head:**
   * Conv2D (Kernel: 1×1, Filters: 1) → BatchNorm → ReLU
   * Flatten
   * Dense (Size: 256) → ReLU
   * Dense (Size: 1) → Tanh Activation (Evaluation [-1, +1])

## Parameter Footprint
By defaulting to 64 filters and 4 residual blocks, the parameter count stays under 500,000. This ensures the exported ONNX model remains around 2MB, allowing near-instant network transfer and smooth WebWorker execution.