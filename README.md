# Handwritten-Calculator
Calculates hand written operations using a convolutional neural network trained on the MNIST dataset.

## Introduction
Many handwritten calculators based on the MNIST dataset have been created by amateur machine learning enthusiasts and are available online. However, none follow a rigorous approach to the problem as I have done in the following implementation. This repository will show you step by step how to create such an application: from the data augmentation for the calculator symbols to creating the user interface using PySimpleGUI.

I have split the problem into three parts: creating the dataset, training the model and then finally building the GUI. Feel free to only read the parts that you are interested in.
However, this was written in mind that you read it sequentially, therefore in order to fully understand why things are done the way they are, it is best to read from start to finish.

## 1. Creating the Dataset
The MNIST dataset consists of 60,000 28x28 grayscale images of the digits 0-9, along witha test set of 10,000 images. My first aim was to create a dataset of operations I intended to use in my calculator. The operations I chose to use were: +,-,*,/,(,). The final dataset had to be of comparable size to that in the MNIST datset - so not to bias the trained neural network. If each operation was hand drawn 50 times, then using various transformations we could turn the 50 images to 1500 unique images (these transformation will be discussed later).
