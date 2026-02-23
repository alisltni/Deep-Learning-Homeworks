# Deep Learning Course - Homework Assignments

This repository contains my homework assignments for a Deep Learning course, covering fundamental concepts to advanced architectures in deep learning.

## Course Overview

The assignments progress through key deep learning topics including PyTorch fundamentals, computer vision with CNNs, time series analysis with RNNs, and generative models with autoencoders.

---

## Repository Structure

```
.
├── HW1/
├── HW2/
├── HW3/
├── HW4/
└── README.md
```

---

## Assignment Details

### HW1: Introduction to PyTorch and Google Colab
**Topics:** PyTorch Fundamentals, Tensor Operations, Google Colab Environment

Getting familiar with PyTorch framework and Google Colab as the development environment. This assignment covers basic tensor operations, automatic differentiation, and setting up the deep learning workflow.

---

### HW2: Convolutional Neural Networks and Physics-Informed Learning  
**Topics:** CNN, CIFAR-10, Dropout, Regularization Techniques, PINN, ODE-Constrained Learning  

This assignment explores supervised image classification and physics-informed neural network modeling through two distinct implementations.

#### Notebook 1: CNN on CIFAR-10 with Dropout Regularization
- Implementation of a CNN architecture from scratch  
- Training on the CIFAR-10 dataset  
- Analysis of dropout and its effect on validation performance  
- Study of regularization and overfitting behavior  

#### Notebook 2: Physics-Informed Neural Network for Coupled ODE System
- Implementation of a PINN for a mass-spring dynamical system  
- Embedding physical constraints directly into the loss function  
- Solving coupled differential equations using neural networks  
- Comparison with numerical ODE solver results  

---

### HW3: Advanced Computer Vision Architectures
**Topics:** Semantic Segmentation, Face Recognition, U-Net, MobileFaceNet

This assignment explores two important computer vision tasks through state-of-the-art architectures.

#### Notebook 1: U-Net for Image Segmentation
- Implementation of U-Net architecture
- Semantic segmentation task

#### Notebook 2: MobileFaceNet for Face Recognition
- Implementation of MobileFaceNet
- Face recognition and verification
- Efficient mobile-friendly architecture design

---

### HW4: Time Series Prediction and Generative Models
**Topics:** RNN, LSTM, GRU, Autoencoders, VAE, β-VAE, CVAE

This assignment covers sequential data modeling and generative modeling approaches.

#### Notebook 1: Time Series Forecasting - Oil Price Prediction
- Dataset: Yahoo Finance oil price data
- Implemented models: RNN, GRU, LSTM
- Comparative analysis of recurrent architectures

#### Notebook 2: Autoencoders and Variational Autoencoders
- Standard Autoencoder (AE) implementation
- Variational Autoencoder (VAE)
- β-VAE for disentangled representations
- Conditional VAE (CVAE) for controlled generation
- Analysis of latent space representations

## Getting Started

Each homework folder contains Jupyter notebooks that can be run directly in Google Colab or locally with Jupyter.

### Running in Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com/)
2. Upload the desired notebook
3. Ensure GPU runtime is enabled: `Runtime > Change runtime type > GPU`
4. Run all cells

### Running Locally
```bash
# Install dependencies
pip install torch torchvision numpy matplotlib pandas jupyter

# Launch Jupyter
jupyter notebook
```

---

## 📊 Key Learnings

- Building neural networks from scratch using PyTorch
- Understanding convolutional operations and their role in feature extraction
- Implementing regularization techniques to prevent overfitting
- Working with advanced architectures like U-Net and MobileFaceNet
- Processing sequential data with RNNs, LSTMs, and GRUs
- Understanding generative modeling through autoencoders and VAEs
- Analyzing and visualizing model performance
