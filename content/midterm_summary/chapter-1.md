---
date: '2025-03-10T12:39:55-04:00'
draft: false
title: 'Chapter 1'
menus: main
---

{{< mathjax >}}

# Neural Network Fundamentals

## Multi-layered Neural Network
- **Structure**:
  - Input: 10 numbers
  - Output: 4 decisions or predictions
- **Backpropagation**:
  - Used to calculate the error in predictions and adjust the network.

## Sensitivity in Neural Networks
- Sensitivity measures how much the error at the output (\(\Delta E\)) changes due to small input variations.
- Sensitivity is propagated backward using:
  $$
  \delta_{p1} = \frac{\partial E}{\partial p1}
  $$
- Error computation block:
  ```math
  \{2(o1 - t1), 2(o2 - t2), 2(o3 - t3), 2(o4 - t4)\}
  ```

## Activation Functions
### Sigmoid Function
```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```
- Outputs range: (0,1)
- **Issues**:
  - Saturates and kills gradients.
  - Not zero-centered.
  - Computationally expensive.

### Tanh Function
```math
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```
- Outputs range: (-1,1)
- Zero-centered but still prone to gradient saturation.

### ReLU (Rectified Linear Unit)
```math
f(x) = \max(0, x)
```
- Does not saturate in the positive region.
- Computationally efficient.
- **Drawbacks**:
  - Can lead to "dead neurons" (never activating).
  - Not zero-centered.

### Leaky ReLU
```math
f(x) = \max(\alpha x, x), \quad \alpha \approx 0.01
```
- Addresses dead neuron issue by allowing small negative values.

### ELU (Exponential Linear Unit)
```math
f(x) = \begin{cases}
x, & x > 0 \\
\alpha (e^x - 1), & x \leq 0
\end{cases}
```
- Zero-centered.
- More robust than ReLU in noise-prone settings.

### Maxout
```math
f(x) = \max(w_1^T x + b_1, w_2^T x + b_2)
```
- Generalizes ReLU and Leaky ReLU.
- Doubles parameters per neuron.

### Practical Choice
- Use **ReLU** as the default.
- Try **Leaky ReLU, Maxout, or ELU**.
- Avoid **Sigmoid**.

## Data Preprocessing
### Zero Mean
```math
X \leftarrow X - \text{mean}(X, \text{axis}=0)
```

### Normalization
```math
X \leftarrow \frac{X}{\text{std}(X, \text{axis}=0)}
```

### PCA & Whitening
- PCA decorrelates data.
- Whitening scales dimensions by eigenvalues.

### Key Rule
- Compute statistics **only on training data**.
- Apply transformation to test/validation data.

## Weight Initialization
- Proper initialization prevents gradient vanishing or explosion.
```math
w = \frac{\text{randn}(n)}{\sqrt{n}}
```
- For ReLU:
```math
w = \frac{\text{randn}(n)}{\sqrt{n/2}}
```

## Batch Normalization (BN)
- **Reduces internal covariate shift**.
- BN normalizes activations:
```math
\hat{x} = \frac{x - \mu}{\sigma}
```
- Applied before the activation function.
- At test time, uses fixed mean/std from training.

## Regularization
### L2 Regularization (Weight Decay)
```math
R(f) = \lambda ||w||^2
```
### L1 Regularization
```math
R(f) = \lambda ||w||
```
- Leads to sparse weights.
### Elastic Net
- Combination of L1 and L2.

### Dropout
- Randomly drops neurons during training.
- **Test-time**: Scale weights by dropout probability \( p \).

## Hyperparameter Optimization
- **Random Search** > Grid Search.
- Bayesian Optimization speeds up the process.
- **Common Hyperparameters**:
  - Learning rate.
  - Batch size.
  - Number of layers.
  - Dropout rate.

## Loss Functions
### Classification
- **Softmax with Cross-Entropy Loss**:
```math
L = - \sum_{i} y_i \log(\hat{y}_i)
```
- **Hinge Loss** (for SVMs):
```math
L = \sum_{i} \max(0, 1 - y_i \hat{y}_i)
```

### Regression
- **L2 Loss**:
```math
L = \sum_i (y_i - \hat{y}_i)^2
```
- **L1 Loss**:
```math
L = \sum_i |y_i - \hat{y}_i|
```

## Parameter Update
- After computing gradients via **backpropagation**, parameters are updated using:

### Vanilla SGD
```math
w \leftarrow w - \eta \nabla w
```

### Momentum Update
```math
v \leftarrow \beta v - \eta \nabla w
```
```math
w \leftarrow w + v
```

### Nesterov Accelerated Gradient
```math
v \leftarrow \beta v - \eta \nabla w'
```
```math
w' \leftarrow w + \beta v
```

## Final Notes
- **Choose activation functions wisely** (ReLU recommended).
- **Use normalization and proper weight initialization**.
- **Apply regularization techniques** (L2, Dropout).
- **Optimize hyperparameters systematically** (Bayesian search preferred).
- **Understand loss functions** based on classification vs. regression.
