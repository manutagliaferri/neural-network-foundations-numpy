# Neural Network Foundations & Analytical Backpropagation (NumPy)

A hands-on implementation of the core mathematical building blocks of neural networks using **pure Python and NumPy**, without deep learning frameworks (no PyTorch, TensorFlow, or autograd engines).

Developed as part of the **Machine Learning coursework at USI**, this project focuses on understanding and coding the internal mechanics of backpropagation, tensor calculus for gradient computation, and training dynamics.

---

## 🧠 What I Implemented

Working within a modular layer-based architecture, I implemented the forward and backward passes:

- **Vectorized Linear Layer:**
  - Forward pass ($Y = XW + b$).
  - Analytical gradients via matrix calculus:
    - $\frac{\partial L}{\partial W} = X^T \cdot \delta$
    - $\frac{\partial L}{\partial b} = \sum_{batch} \delta$
    - $\frac{\partial L}{\partial X} = \delta \cdot W^T$
- **ReLU Activation:** Forward thresholding and sub-gradient backpropagation.
- **Softmax + Categorical Cross-Entropy:**
  - Numerically stable softmax using the max-shift trick.
  - Efficient combined backward pass: $\frac{\partial L}{\partial Z} = \frac{1}{N} (S - T)$.
- **Optimization & Pipeline:**
  - Mini-batch gradient descent with $L_2$ Regularization (Weight Decay).
  - Validation split, evaluation routines, and **Early Stopping** based on validation accuracy patience.

*Note: Numerical gradient checking (finite difference quotients) was used to verify the mathematical correctness of all analytical gradients.*

---

## 📈 Benchmarks

1. **Synthetic Two-Spirals:**
   - Non-linear binary classification using a 3-layer MLP (`2 -> 50 -> 30 -> 2`).
   - Converged to a clean non-linear decision boundary with loss $< 0.033$.

2. **MNIST Digit Recognition:**
   - 2-layer network (`784 -> 20 -> 10`) trained on 42,000 samples and evaluated on 10,000 test images.
   - Achieved **96.08% Test Accuracy** with early stopping triggered at epoch 80.

---

## 📂 Structure

```text
├── neural_network_foundations.ipynb
└── README.md
