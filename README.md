# Handwritten-Calculator
Calculates hand written operations using a convolutional neural network trained on the MNIST dataset.

## Introduction
Many handwritten calculators based on the MNIST dataset have been created by amateur machine learning enthusiasts and are available online. However, none follow a rigorous approach to the problem as I have done in the following implementation. This repository will show you step by step how to create such an application: from the data augmentation for the calculator symbols to creating the user interface using PySimpleGUI.

I have split the problem into three parts: creating the dataset, training the model and then finally building the GUI. Feel free to only read the parts that you are interested in.
However, this was written in mind that you read it sequentially, therefore in order to fully understand why things are done the way they are, it is best to read from start to finish.

## 1. Creating the Dataset
The MNIST dataset consists of 60,000 28x28 grayscale images of the digits 0-9, along with a test set of 10,000 images. My first aim was to create a dataset of operations I intended to use in my calculator. The operations I chose to use were: +, -, *, /, (, ). The final dataset had to be of comparable size to that in the MNIST datset - so not to bias the trained neural network. 

If each operation was hand drawn 50 times, then using various transformations (discussed later) we could turn the 50 images to 1500 unique images of each operation. This would mean we could repeat each unique image just 4 times in order to get a somewhat similar representation of each operation to the representation of each number in the MNIST dataset. In my opinion repeating each image 4 times was acceptably low. Considering the final accuracy of the trained model on the test set was 0.9921, the time required to hand draw more operations would not be worth the marginal increase in accuracy.

The first transformation applied to the dataset is a pixel inversion. This is not to increase the number of unique examples but by inverting the pixel intensities we improve performance due to data centering, (the mean is close to 0). This is explained further in: https://stats.stackexchange.com/questions/220164/impact-of-inverting-grayscale-values-on-mnist-dataset.

The second set of transformations applied to the dataset are pixel shifts. This is the first set of transformations used in order to increase the number of unique images.
Each of the original images is shifted by 1 pixel up, down, left and right. This results in the original dataset of 50 hand drawn images becoming 250 images (including the unshifted images).

The next transformation is vertically flipping every image. Since all the operations should be vertically symmetric, by flipping them along the horizontal axis we can double the number of images from 250 to 500.

The last transformation I performed was a rotation. Each image was rotated 10 degrees clockwise and 10 degrees anticlockwise. This tripled the number of images (inlcuding the unrotated images).

![Diagram of Transformations](Images/Transformations.png "Diagram of Transformations")
*Visualisation of Transformations*

The final step was to repeat this dataset 4 times in order for a similar representation of the operations as the numbers in the MNIST dataset, join the two datasets and then shuffle them.
