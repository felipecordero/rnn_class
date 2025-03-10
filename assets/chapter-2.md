### **Midterm Exam Summary: Probabilistic Graphical Models (PGMs)**

---

## **1. Introduction to Probabilistic Graphical Models (PGMs)**
- PGMs provide a **declarative representation** of our understanding of the world.
- **Key Benefits**:
  - Can handle **uncertainty** (partial/noisy observations, missing data).
  - **Separate knowledge from reasoning**—avoiding special-purpose algorithms.
  - Allow for different **inference algorithms** on the same model.

### **Types of PGMs:**
1. **Directed Graphs (Bayesian Networks)**
2. **Undirected Graphs (Markov Random Fields)**

---

## **2. Bayesian Networks (BNs)**
- **Definition**: A Bayesian Network is a **Directed Acyclic Graph (DAG)** that provides a compact factorized representation of a joint probability distribution.
- **Components**:
  - **Nodes**: Represent **random variables**.
  - **Edges**: Represent **causal influences**.
  - **Factorization** follows the **chain rule**:
    ```math
    P(X_1, X_2, ..., X_n) = \prod_{i} P(X_i | Parents(X_i))
    ```

### **Bayesian Networks - Example**
- A student's grade depends on **intelligence, course difficulty, SAT score, and recommendation letter**.
- Example of a Conditional Probability Table (CPT):
  ```math
  P(X_n | Z_n = k) = P(X_n; \mu_k, \Sigma_k)
  ```

### **Inference in Bayesian Networks**
- **Exact Inference**:
  - **Marginalization**: Summing over all possible values of hidden variables.
    ```math
    P(Y = y | E = e) = \frac{P_B(y, e)}{P_B(e)} = \frac{P_B(y, e)}{\sum_{y} P_B(y, e)}
    ```
  - **Conditioning**: Given a variable \( g_1 \), find \( P_B(I, D | g_1) \).

- **Reasoning Patterns**:
  - **Causal reasoning (Prediction)**: Finding probability of an effect given the cause.
  - **Evidential reasoning (Explanation)**: Inferring cause given the effect.
  - **Inter-causal reasoning**: Finding relations between multiple influencing factors.

---

## **3. Conditional Independence in BNs**
- **Definition**: Two variables are **conditionally independent** given a third variable.
  - Example: **Rain (A) and Sprinkler (B) are conditionally independent given Wet Grass (C)**.

- **Common Cause & Common Effect**:
  - **Common Cause**: Two variables share a common cause but don’t influence each other directly.
  - **Common Effect**: Two variables independently affect a third variable.

---

## **4. Naïve Bayes Model**
- **A simplified Bayesian network for classification**:
  ```math
  P(C | X_1, X_2, ..., X_n) = \frac{P(C) \prod_i P(X_i | C)}{P(X_1, X_2, ..., X_n)}
  ```
- Assumes **features are conditionally independent given the class**.

---

## **5. Dynamic Bayesian Networks (DBNs)**
- Extends Bayesian Networks to **model time-dependent data**.
- **Markov Assumption**: The future state depends only on the present state, not past history.
- **2 Time-Slice Bayesian Network (2-TSBN)**:
  - Nodes represent variables across time steps.
  - Used in **speech recognition, robotics, finance**.

---

## **6. State-Space Models (SSMs)**
- A model of how state variables evolve over time:
  - **State transition**: \( P(X_t | X_{t-1}) \)
  - **Observation function**: \( P(Y_t | X_t) \)

- **Types of SSMs**:
  - **Hidden Markov Models (HMMs)**:
    - Discrete states 
    ```math
    \( X_t \in \{1, 2, ..., K\} \)
    ```
    - Used in **speech recognition, NLP**.
  - **Kalman Filter Models (KFMs)**:
    - Continuous states \( X_t \in \mathbb{R}^N \).
    - Assumes **Gaussian distributions**.

- **Inference in SSMs**:
  - **Filtering**: Compute current state given past observations.
  - **Smoothing**: Compute past states given future observations.
  - **Prediction**: Compute future states given current state.
  - **Control**: Compute best action to take.
  - **Decoding**: Find most likely sequence of states.
  - **Classification**: Assign class labels.

---

## **7. Markov Random Fields (MRFs)**
- **Undirected graphs**, where the joint distribution is factorized over **maximal cliques**.
  ```math
  P(X) = \frac{1}{Z} \prod_{c \in C} \phi_c(X_c)
  ```
  where:
  - \( Z \) is the **partition function** (normalization).
  - \( \phi_c(X_c) \) are **potential functions**.

- **Properties**:
  1. **Factorization**: Uses non-negative potential functions.
  2. **Conditional Independence**: Nodes in **A and B are conditionally independent given C**:
     ```math
     P(A, B | C) = P(A | C) P(B | C)
     ```

---

## **8. Image De-Noising using MRFs**
- **Problem**: Remove noise from an image where pixel values flip with 10% probability.
- **Solution**:
  - Assume **neighboring pixels are correlated**.
  - Construct an MRF with two types of cliques:
    - **\( \{x_i, y_i\} \) pairs**
    - **\( \{x_i, x_j\} \) neighboring pixels**
  - Define **energy function**:
    ```math
    E(X, Y) = h \sum_i x_i - \beta \sum_{\{i,j\}} x_i x_j - \eta \sum_i x_i y_i
    ```
  - Compute probability using:
    ```math
    P(X) \propto \exp(-E(X, Y))
    ```

---

## **9. Summary of Key Equations**
- **Bayesian Network Factorization**:
  ```math
  P(X_1, ..., X_n) = \prod_{i} P(X_i | Parents(X_i))
  ```
- **Exact Inference**:
  ```math
  P(Y = y | E = e) = \frac{P_B(y, e)}{\sum_{y} P_B(y, e)}
  ```
- **Markov Random Fields Joint Distribution**:
  ```math
  P(X) = \frac{1}{Z} \prod_{c \in C} \phi_c(X_c)
  ```
- **Image Denoising Energy Function**:
  ```math
  E(X, Y) = h \sum_i x_i - \beta \sum_{\{i,j\}} x_i x_j - \eta \sum_i x_i y_i
  ```

---

### **Final Thoughts**
- **BNs** model causal dependencies, **MRFs** model relationships without causality.
- **DBNs** extend BNs to **time-dependent problems**.
- **State-Space Models** capture **hidden states** and observations over time.
- **MRFs are used in computer vision, speech recognition, and NLP**.
