
# AMGNN - A Minimal Deep Learning Framework from Scratch

AMGNN (Amine's Minimal General Neural Network) is a C++ deep learning framework developed purely from scratch, without relying on external machine learning libraries. It was built with the goal of deeply understanding how neural networks, backpropagation, optimization, and training pipelines work under the hood.

> ⚠️ This project is not intended to compete with TensorFlow or PyTorch, but to serve as a **learning resource** and **low-level educational framework** for those interested in how AI works at its core.

---

## 🚀 Features

- ✅ Fully custom implementation of:
  - Neural networks (NN)
  - Convolutional Neural Networks (CNN)
  - Backpropagation
  - Forward propagation
  - Loss functions (MSE, Binary Cross Entropy, etc.)
  - Optimizers (Gradient Descent, Mini-Batch, Stochastic, L-BFGS, Adam, AMGO)
  - Activation functions
  - Numerical differentiation using interpolation-based methods

- 🧠 Focused on understanding:
  - Matrix operations
  - Derivatives
  - Optimizer behavior
  - Training dynamics without hardware acceleration

- 🎯 Written in pure C++ with OpenCV used only for image input/output.

---

## 🧩 Structure

```
AMGNN/
│
├── Layers/               # Fully connected, convolutional, and more
├── Activations/          # Sigmoid, ReLU, Softmax, etc.
├── LossFunctions/        # MSE, CrossEntropy, etc.
├── Optimizer/            # Custom optimization methods (SGD, Adam, L-BFGS)
├── Differentiation/      # Numerical gradient calculation
├── AMGNN.cpp             # Main engine and training API
├── Utils/                # Image reader, helper functions
└── main.cpp              # Example usage and test code
```

---

## 🔧 Dependencies

- **C++17 or later**
- **OpenCV** for image processing (only used in image loading)

---

## 🧪 Example Usage

```cpp
#include "AMGNNv1.2/AMGNN.cpp"

// Example for loading and training a CNN
std::vector<cv::Mat> inputs = AMGNN::readImage(...); // normalized to [0,1]
std::vector<std::vector<double>> labels = ...;

CNN cnn;
AMGNN model(cnn, Loss_Function::CROSS_ENTROPY, Optimizers::ADAM);
model.train(inputs, labels);
```

---

## 📚 Educational Purpose

This project was built as a personal challenge to:

- Learn how to implement a neural network from first principles
- Understand all the mathematics and memory management
- Write an end-to-end training loop with gradient-based optimization
- Get familiar with numerical stability and low-level issues

If you're interested in how frameworks like PyTorch work under the hood, this project is a valuable resource.

---

## 📝 Future Work

Although no longer actively developed, ideas for future improvements (for educational exploration):

- Port to Python using NumPy
- Add visualization tools for training stats
- Improve memory management and performance
- Extend to include RNNs or Transformer modules

---

## ⚠️ Disclaimer

> This framework is **not optimized** for real-world usage or large-scale training. It lacks GPU acceleration and advanced parallelism, and is intended for **learning purposes only**.

---

## 📜 License

This project is open-source and available under the MIT License. Contributions, forks, and educational use are welcome!

---

## 🙏 Acknowledgments

Special thanks to:
- All the amazing tutorials, YouTube channels, and open-source developers that inspired this project.
- OpenCV for image processing support.
- Anyone who takes the time to learn from this code.

---

## 💬 Feedback

If you find this project interesting or useful, feel free to ⭐ the repo, open issues, or ask questions.

---

> Made with patience, frustration, and curiosity by **Amine Guettara** 💡
