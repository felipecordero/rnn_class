# Midterm Exam Summary

## Neural Networks Overview
- Neural networks are **universal approximators** that can model Boolean, categorical, or real-valued functions.
- Capable of pattern recognition in static inputs, time-series, and sequences.
- Training is required for prediction.
- Neural networks consist of multiple layers:
  - **Input Layer**: Accepts raw data.
  - **Hidden Layers**: Extracts features through weighted connections.
  - **Output Layer**: Produces final prediction or classification.
- Common activation functions:
  - **Sigmoid**: $ \sigma(x) = \frac{1}{1 + e^{-x}} $ (good for probabilities)
  - **ReLU**: $ f(x) = \max(0, x) $ (reduces vanishing gradient problem)
  - **Tanh**: $ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $ (zero-centered outputs)

## 1-D Non-Linearly Separable Data
- Consider a one-dimensional dataset where class labels cannot be separated by a simple threshold.
- **Estimating probability**: The probability of $Y=1$ at a given point is computed using a small surrounding window.
- The goal is to find a function that can model $P(Y=1 | X)$. 

## Estimating the Model
- A **sigmoid perceptron** with a single input can approximate the posterior probability of a class given input:
  $$ P(Y=1 | X) = \sigma(WX + b) $$
- Training is performed using **gradient descent** to optimize parameters $W$ and $b$.

## Maximum Likelihood Estimation (MLE)
- MLE minimizes the **KL divergence** between the desired and actual output.
- The loss function is derived from likelihood principles:
  $$ L(\theta) = -\sum_{i} y_i \log P(Y_i | X_i, \theta) + (1 - y_i) \log(1 - P(Y_i | X_i, \theta)) $$
- Requires **gradient descent** for optimization.

## Network Structure
- The final neuron in the network acts as a **linear classifier** on the transformed data.
- The network **transforms non-linear data** into linearly separable feature space.
- **Feature extractor layers** convert input space $X$ into feature space $Y$ where classes are separable.
- Feature extraction layers include **convolutional layers (CNNs), recurrent layers (RNNs), and transformer layers**.

## Autoencoder (AE)
- **Definition**: A neural network that copies its input to its output.
  - **Encoder**: $h = f(x)$
  - **Decoder**: $r = g(h)$
- Used for **dimensionality reduction** and feature extraction.
- Applications:
  - Image compression
  - Feature extraction for classification
  - Denoising corrupted signals

## Linear vs. Non-Linear Autoencoders
- Linear AE minimizes squared reconstruction error (like PCA).
- Non-Linear AE finds a **non-linear manifold** that best represents the data.
- Loss function for reconstruction:
  $$ L(x, g(f(x))) = || x - g(f(x)) ||^2 $$

## Deep Autoencoder
- **Multi-layer Autoencoders** capture more complex structures.
- Trained using **Restricted Boltzmann Machines (RBMs)** followed by **fine-tuning with backpropagation**.
- **Layer-wise pretraining** is often used to initialize deep autoencoders before fine-tuning.

## Avoiding Trivial Identity Mapping
### Undercomplete Autoencoders
- Restrict $h$ to lower dimension than $x$.
- Loss function:
  $$ L(x, g(f(x))) $$
- Non-linear encoders extract useful features beyond PCA.

### Overcomplete Autoencoders
- Higher dimensional $h$ may lead to trivial identity mapping.
- Regularization techniques:
  - **Sparse AE**: Encourages sparsity.
  - **Denoising AE**: Recovers corrupted input.
  - **Contractive AE**: Encourages robustness to perturbations.

## Variational Autoencoder (VAE)
- Imposes a constraint on the latent variable $z$.
- Ensures $z$ follows a **Gaussian distribution**:
  $$ p(z) = \mathcal{N}(0, I) $$
- Uses **reparameterization trick** to enable backpropagation:
  $$ z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$

## VAE Loss Function
- Combination of **reconstruction loss** and **KL divergence loss**:
  $$ L = \mathbb{E}[\log p(x|z)] - D_{KL}(q(z|x) || p(z)) $$
- KL divergence ensures that the latent variable follows a Gaussian distribution:
  $$ D_{KL}(q(z|x) || p(z)) = \frac{1}{2} \sum \left( \sigma^2 + \mu^2 - 1 - \log \sigma^2 \right) $$

## Gaussian Mixture Models (GMM)
- Used for modeling complex distributions as a combination of Gaussian distributions.
- Objective: Learn **means, variances, and mixing probabilities**.
- EM Algorithm is commonly used for training GMMs.

## Generative Models
- Neural networks can be used as **generative models** to learn probability distributions.
- Examples:
  - Variational Autoencoders (VAEs)
  - Generative Adversarial Networks (GANs)
  - Diffusion Models
- Applications include **image generation, text synthesis, and semantic hashing**.

## Applications of Autoencoders
1. **Dimensionality reduction**
2. **Feature learning for classification**
3. **Image retrieval using binary codes** (semantic hashing)
4. **Data denoising**
5. **Anomaly detection** in cybersecurity and healthcare

## Conclusion
- Autoencoders provide powerful tools for representation learning.
- VAEs extend AE capabilities by enabling **probabilistic data generation**.
- Regularization techniques ensure meaningful feature extraction.
- **Generative models** play a crucial role in modern AI applications.

