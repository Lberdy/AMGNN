# AMGNN
# AMGNN – Amine Guettara Neural Network
Hi! This is AMGNN, my custom neural network framework built entirely in C++ (from scratch).

Current Version : 1.1.2

My Email : guettaraamine2@gmail.com

Features

     Optimization Methods:
        Gradient Descent
        Mini-Batch Gradient Descent
        Stochastic Gradient Descent
        L-BFGS
        
     Optimizers:
        AMGNNO – Supports multiple learning rate decay functions (step, exponential, etc.)
        ADAM
        L-BFGS Optimizer (Quasi-Newton method)
        
     Differentiation:
        Uses interpolational polynomial methods to calculate gradients
        Supports three differentiation orders: 2, 4, and 6
        
     Task Types Supported:
        Regression
        Multiclass Classification
        Binary Classification
        Multilabel Classification
        
     Multithreaded Training:
        Threading for weights
        Threading for batches
        Thread usage is fully configurable
        Efficient Thread Pooling to adapt to your CPU
        
     Model Persistence:
        Save and load your models easily — portable and ready for reuse
        
 Limitations (for now)
 
    No GPU acceleration yet — planned for the future once GPU access is available (CUDA/OpenCL support coming soon)
    
 Coming Soon
 
     Convolutional Layer – postponed until I get a GPU... my CPU is crying already 😅
     Full Documentation covering the framework’s internal structure, setup, and examples

