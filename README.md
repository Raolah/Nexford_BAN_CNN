# Nexford_BAN_CNN


## README

## FASHION MNIST CLASSIFICATION

This is a project assigned to classify images using profile to target marketing for different products by working with the Fashion MNIST dataset in Keras and also adapt the code for user profile classification.


## INSTRUCTIONS

We will perform the following tasks:

1.	1.	Convolutional Neural Network (CNN):

o	Using Keras and classes in both Python and R, develop a CNN with six layers to classify the Fashion MNIST dataset.


2.	Prediction:

o	Make predictions for at least two images from the Fashion MNIST dataset.



## INSTALLATION

Install tensorflow in python
```bash
pip install tensorflow  numpy matplotlib
```

Install keras in R
```bash
install.packages("keras")
install.packages("tensorflow")
```
## Running Python


#Steps To Using Keras and Classes in Python:

Import all libraries
```bash
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
```

Load the dataset from Fashion MNIST dataset
```bash
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
```

Build the CNN model
```bash
model = models.Sequential()
```

Make Predictions
```bash
predictions = model.predict(test_images)
```

#Steps To Using Keras and Classes in R:

Import all libraries
```bash
library(keras)
library(ggplot2)
```

Load the dataset from Fashion MNIST dataset
```bash
fashion_mnist <- dataset_fashion_mnist()
```

Build the CNN model
```bash
model <- keras_model_sequential() %>%
```

Make Predictions
```bash
predictions <- model %>% predict(test_images)
## Lessons  

To import the Fashion MNIST dataset in Keras into Python and R we have to install all neccessary libraries for both Python and R, then import all libraries including tensorflow, matlotlib, plyplot and numpy for python, keras and tensorflow for R.

For the excecution of code in python and R, load the dataset then preprocess the data to normalize the images to a range of 0 to 1 by dividing by 255. Then we add a channel for images to reshape the data.

After that we then buils and train the CNN model then compile and train the model. So in order to make predictions we have to evaluate the model then create a visualization to display the prediction of the two images.


## Requirements

* Python 3.x
* TensorFlow
* Keras
* R with keras and tensorflow packages

