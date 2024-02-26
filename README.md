# CIFAR-10 Image Classification using Convolutional Neural Network (CNN)

## Overview

This project aims to build a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 color images of size 32 x 32 distributed over 10 classes, making it a widely used benchmark in research for image classification tasks.

## Steps to Reproduce

### 1. Load and Normalize CIFAR-10 Dataset

Download the CIFAR-10 dataset from the official website: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The dataset is split into 50,000 training images and 10,000 test images. Normalize the pixel values of the images.

### 2. Define CNN Model, Loss Function, and Optimizer

Define a Convolutional Neural Network (CNN) model suitable for image classification. Choose an appropriate loss function and optimizer for training the model. Consider using techniques such as dropout to reduce overfitting.

### 3. Data Augmentation

Implement data augmentation using `torchvision.transforms` to artificially increase the diversity of the training dataset. This helps in improving the generalization ability of the model.

### 4. Training the Model

Train the CNN model on the training dataset. Monitor training and validation loss and accuracy during the training process.

### 5. Evaluation

Test the trained model on the test dataset and report the test accuracy. Evaluate the model's performance on unseen data.

### 6. Plotting Model Curves

Plot the training and validation curves for both loss and accuracy. Visualize how these metrics change over epochs to analyze the model's learning progress.

## Requirements

- Python (>=3.6)
- PyTorch (>=1.0)
- torchvision
- matplotlib

## License

This project is licensed under the [MIT License](LICENSE).

