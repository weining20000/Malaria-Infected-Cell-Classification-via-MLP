# Malaria-Infected-Cell-Classification-via-MLP
## Requirements
* PyCharm
* Python 2.7/3.5
* Google Cloud Platform
* Keras
* Multi-Layer Perceptron

## Introduction
This project built a Multi-Layer Perception (“MLP”) model to classify the cells into 4 different categories, which are: “red blood cell”, “ring”, “schizont”, and “trophozoite.” The best model in the seven-day data challenge competition achieved a reasonable performance with a 0.6147 macro-averaged F1-score and 0.6212 Cohen’s Kappa Score.

## Dataset
The original dataset (“the Train dataset”) contains 8,607 cell images of size 100 X 103 and 8,607 txt files recording the corresponding string label for each cell image. Of the 8,607 cell images, 7,000 are red blood cell, 365 are ring, 133 are schizont, and 1,109 are trophozoite. Since the dataset is unbalanced and the raw data (.png and .txt) cannot be directly used to train the network, the following steps were performed to preprocess the dataset.

## Data preprocessing
* Image Resizing
* Data Augmentation
* Labeling <br />
(Detailed description of the data preprocessing steps are provided in the project report under the **Report** repository.)

## Modeling
The model used in this project was a three-layer MLP network. The input dimension is 7,500 (50 X 50 X 3). The number of neurons for each layer is 128, 32, and 32, respectively. The activation functions used in the hidden layers are Regular Linear Unit (“Relu”) function, while the output layer adopts the Softmax activation function, which is particularly useful for calculating the probability distribution of multi-class outputs. The optimizer is Adam, and the loss function is Categorical Crossentropy, a function that specifically targeted at categorical outputs. Lastly, a 10-fold cross-validation was implemented to test the effectiveness of the model.  

