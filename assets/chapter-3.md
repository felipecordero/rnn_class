# Chapter 3: Deep Belief Network (DBN)

## 1. Introduction to Deep Belief Networks (DBN)
- **Deep Belief Networks (DBNs)** are generative models composed of **multiple layers of hidden units**.
- They are constructed using **Restricted Boltzmann Machines (RBMs)**, which are stacked to form a deep structure.
- **RBMs** serve as the building blocks of DBNs and are a special type of **Boltzmann Machines (BMs)**, which belong to the family of **Markov Random Fields (MRFs)**.

### **Boltzmann Machines (BMs)**
- A **Boltzmann Machine** is a **stochastic neural network** consisting of symmetrically connected binary units.
- It is used to **learn probability distributions** over data.

### **Restricted Boltzmann Machines (RBMs)**
- RBMs are a **special case of Boltzmann Machines** with a restricted architecture:
  - **No intra-layer connections** (i.e., no connections between neurons in the same layer).
  - Composed of:
    - **Visible units**: $ v \in \{0,1\}^D $
    - **Hidden units**: $ h \in \{0,1\}^D $

### **Why Use DBNs?**
- DBNs solve the **difficulty in training deep neural networks** by:
  1. **Pretraining** each layer independently as an RBM.
  2. **Using a greedy layer-wise training approach** to initialize deep networks efficiently.
  3. **Improving a lower bound on the likelihood** with each additional layer.

---

## 2. Restricted Boltzmann Machine (RBM)

### **Structure of an RBM**
- RBMs consist of:
  - A **visible layer** containing observed data.
  - A **hidden layer** capturing latent features.
  - A **weight matrix** \( W \) connecting visible and hidden units.
  - **Bias terms** \( a \) and \( b \) for visible and hidden layers, respectively.

### **Energy Function of an RBM**
The **energy function** for an RBM is defined as:

$$
E(v, h) = - v^T W h - a^T v - b^T h
$$

where:
- \( W \) is the weight matrix.
- \( a \) is the bias vector for visible units.
- \( b \) is the bias vector for hidden units.

### **Probability Distributions in RBMs**
- The **joint probability distribution** is given by:

$$
P(v, h) = \frac{1}{Z} \exp(-E(v, h))
$$

where \( Z \) is the **partition function**:

$$
Z = \sum_{v,h} \exp(-E(v, h))
$$

- The **marginal probability of visible units**:

$$
P(v) = \sum_h P(v, h)
$$

---

## 3. Inference in DBNs

### **Exact Inference**
- Exact inference is generally **intractable** because:
  - The **latent space is high-dimensional**.
  - The **posterior distribution is complex** and difficult to compute directly.

### **Approximated Inference (Sampling)**
- To estimate expectations efficiently, **sampling methods** are used:
  1. **Basic Sampling - Transformation Method**
     - Generates samples using a transformation function.
  2. **Basic Sampling - Rejection Sampling**
     - Uses a proposal distribution to generate samples and rejects unlikely ones.
  3. **Gibbs Sampling**
     - A **Markov Chain Monte Carlo (MCMC)** method where each variable is sampled conditionally.

#### **Gibbs Sampling for RBMs**
- Used to approximate \( P(v) \).
- Each step involves:
  - Updating one variable \( z_i \) while keeping others fixed:

  $$
  p(z_i | z_{\setminus i})
  $$

- **Example Process**:
  - Start with an initial state \( (v_0, h_0) \).
  - Iteratively update the state using conditional probabilities.

---

## 4. Learning in RBMs

### **Maximum Likelihood Learning**
- Learning is done by maximizing the **likelihood function**:

$$
\max_{w_{ij}, a_i, b_j} \sum_{l=1}^{m} \log \sum_{h} P(v^l, h^l)
$$

- The gradient of the **log-likelihood**:

$$
\frac{\partial \log P(v)}{\partial W_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}
$$

- **Contrastive Divergence (CD) Algorithm**:
  - A **sampling-based approximation** for computing the gradient efficiently.

---

## 5. Deep Belief Networks (DBN)

### **Greedy Layer-Wise Training**
- DBNs use a **layer-wise** training approach:
  1. Train the **first RBM** using input data.
  2. Use the activations from the trained RBM as input to the **next RBM**.
  3. Repeat the process for **each subsequent layer**.

### **DBN Feature Generation**
- A DBN can **extract hierarchical features** from data.
- Example:
  - **Architecture**: 784-1000-500-250-30
  - Each layer learns a **higher-level representation**.

### **DBN Applications**
- **Feature Learning**
- **Dimensionality Reduction**
- **Image Reconstruction**
- **Classification Tasks** (e.g., fingerprint recognition)

---

## 6. Classification Using DBNs

- DBNs can be used for **image classification** and **biometric recognition**.
- **Example**: Fingerprint **liveness detection**.
  - Determines whether a fingerprint is **real or fake** before recognition.

### **Advantages of DBNs in Classification**
- **Efficient training using pretraining**
- **Captures deep feature representations**
- **Generalizes well across different datasets**

---

## Summary of Key Equations

| Concept | Equation |
|---------|----------|
| **Energy Function** | $$ E(v, h) = - v^T W h - a^T v - b^T h $$ |
| **Joint Probability** | $$ P(v, h) = \frac{1}{Z} \exp(-E(v, h)) $$ |
| **Partition Function** | $$ Z = \sum_{v,h} \exp(-E(v, h)) $$ |
| **Marginal Probability** | $$ P(v) = \sum_h P(v, h) $$ |
| **Gradient Update for Learning** | $$ \frac{\partial \log P(v)}{\partial W_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}} $$ |

---

## Conclusion
This chapter covers **Deep Belief Networks (DBNs)** and their foundation in **Restricted Boltzmann Machines (RBMs)**. The key takeaways are:
- **RBMs serve as the building blocks of DBNs**.
- **DBNs use a layer-wise training approach**.
- **Learning in RBMs is done using Maximum Likelihood and Contrastive Divergence**.
- **DBNs are widely used in feature extraction, dimensionality reduction, and classification**.

---

This markdown file now contains **expanded details, complete equations, and formatted LaTeX math expressions**. Save it as `dbn_summary.md` and open it in a markdown viewer with LaTeX support for a well-formatted document. 🚀
