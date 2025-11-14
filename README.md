Deep Learning Image Classification using CNN and Transfer Learning
📌 Project Description

This project focuses on building two deep learning models for image classification:

A Custom Convolutional Neural Network (CNN) built from scratch
A Transfer Learning Model using pre-trained architectures
The objective is to preprocess image data, train both models, compare their performance, and analyze the impact of techniques such as Early Stopping and Learning Rate Reduction.

📂 Dataset

The project uses the CIFAR-10 dataset, containing:
60,000 color images (32×32 pixels)
10 classes with 6,000 images per class
Dataset link: https://www.cs.toronto.edu/~kriz/cifar.html

🧹 Data Preprocessing

Loaded and standardized CIFAR-10 data
Resized & normalized images to scale pixel values to [0, 1]
Applied data augmentation (rotation, flipping, zooming, shifting)
Visualized sample images and class labels
Objective: Ensure consistent image size, reduce overfitting, and improve generalization.

🧱 Model Architecture
🔹 Custom CNN Model

Multiple Convolutional + MaxPooling layers
ReLU activations
Fully connected dense layers
Softmax output layer for classification

🔹 Transfer Learning Models

Fine-tuned pre-trained ImageNet models such as VGG16, ResNet50, or InceptionV3
Frozen base layers + new custom classifier on top
Faster training and improved accuracy with limited data

⚙️ Model Training
Custom CNN

Optimizers used: Adam / SGD
Batch training with augmented data
Early Stopping (Applied):
Monitors validation accuracy
Stops when no improvement for 8 epochs
Automatically saves best weights

Transfer Learning

Fine-tuning selected pre-trained models
Early Stopping applied
Learning Rate Reduction (LRR):
Monitors validation accuracy
Patience: 4 epochs
Reduces LR by factor when performance stalls
Minimum LR set to prevent model collapse

📊 Model Evaluation

Evaluated models on validation/testing datasets
Metrics computed:
Accuracy
Precision
Recall
F1-score
Visualized confusion matrix for class-wise performance
Compared CNN vs Transfer Learning model performance

🔁 Transfer Learning Summary

Implemented several pre-trained models (e.g., VGG16, ResNet, Inception)
Fine-tuned for CIFAR-10 classification

Observations:

Transfer Learning models achieved higher accuracy
Faster convergence due to pre-learned features
More robust for small datasets

📈 Results & Discussion

Custom CNN:

Good performance but required longer training
More prone to overfitting without callbacks

Transfer Learning:

Outperformed CNN in accuracy and speed
Benefited significantly from LR Reduction
More stable training with fewer epochs

Conclusion:
Transfer Learning offers substantial advantages for image classification tasks, especially when computational resources or data are limited.

🧾 Code Quality

Modular, well-structured, and thoroughly commented code

Clear separation of preprocessing, modeling, training, and evaluation phases

Easy for others to follow and reproduce results
## Authors
- Sergei Volkov <taranarmo@gmail.com>
- Mohamad Traiki <m_traiki@gmx.de>
- Mitesh Parab <miteshparab89@gmail.com>
