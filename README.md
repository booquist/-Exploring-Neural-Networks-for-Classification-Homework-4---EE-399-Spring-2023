# -Exploring-Neural-Networks-for-Classification-Homework-4---EE-399-Spring-2023
**Author:** Brendan Oquist <br>
**Abstract:** This report focuses on the application of feedforward neural networks for two distinct tasks as part of EE 399 Spring Quarter 2023, Homework #4. The first task involves fitting a three-layer feedforward neural network to a given dataset, using different sets of training and testing data, and comparing the performance of these models to those developed in Homework #1. The second task entails training a feedforward neural network on the MNIST dataset, following the computation of the first 20 PCA modes of the digit images. The performance of this neural network is compared to Long Short-Term Memory (LSTM), Support Vector Machines (SVM), and decision tree classifiers.

## I. Introduction and Overview
In the first task, the dataset from Homework #1 is revisited, and a three-layer feedforward neural network is employed to fit the data. The performance of the neural network is assessed using the least-square error metric on different training and testing sets. A comparison of the neural network model with the models developed in Homework #1 is provided, highlighting the advantages and disadvantages of using neural networks for this dataset.

In the second task, the MNIST dataset is preprocessed and the first 20 PCA modes of the digit images are computed to perform dimensionality reduction. A feedforward neural network is trained to classify the handwritten digits, and its performance is compared to LSTM, SVM, and decision tree classifiers. The report provides insights into the strengths and weaknesses of these methods for digit recognition tasks and discusses the implications of using feedforward neural networks for image classification problems in the context of machine learning and computer vision.

## II. Theoretical Background
In this section, we provide the necessary mathematical background for fitting feedforward neural networks and analyzing the performance of various machine learning techniques, including Long Short-Term Memory (LSTM), Support Vector Machines (SVM), and decision trees. We also introduce the concept of Principal Component Analysis (PCA) for dimensionality reduction.

### 1. **Feedforward Neural Networks**
A feedforward neural network is a type of artificial neural network where the connections between nodes do not form a cycle. The information moves in only one direction, from the input layer, through hidden layers, to the output layer. The main building block of a neural network is the neuron, which receives input from other neurons, computes a weighted sum, applies an activation function, and passes the result to other neurons. The performance of a neural network depends on the architecture, the weights of the connections, and the choice of activation functions.

In our analysis, we employ a three-layer feedforward neural network to fit the given dataset and to classify handwritten digits from the MNIST dataset.

### 2. **Principal Component Analysis (PCA)**
Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms the original dataset into a new coordinate system where the axes are the principal components. These components are linear combinations of the original features and are chosen in such a way that they are uncorrelated and capture the maximum variance in the data.

The first step in PCA is computing the covariance matrix of the data. The eigenvectors and eigenvalues of the covariance matrix are then computed, and the eigenvectors corresponding to the largest eigenvalues are selected as the principal components. The data is then projected onto the subspace spanned by these components, effectively reducing the dimensionality while retaining the most significant information.

In our analysis, we compute the first 20 PCA modes of the MNIST digit images as part of the preprocessing step.

### 3. **Long Short-Term Memory (LSTM)**
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture that is designed to learn long-term dependencies in sequences of data. LSTMs have a unique structure, including memory cells and multiple gates, which enable them to remember and forget information selectively. The LSTM cell consists of an input gate, a forget gate, and an output gate, which together determine the flow of information through the cell.

In our analysis, we compare the performance of a feedforward neural network with an LSTM model for classifying handwritten digits from the MNIST dataset.

### 4. **Support Vector Machines (SVM)**
Support Vector Machines (SVM) are supervised learning models used for classification and regression tasks. In the context of classification, SVM aims to find the optimal hyperplane that separates the data points belonging to different classes with the largest margin possible. The data points closest to the hyperplane are called support vectors, and they determine the position of the hyperplane. SVM can also handle non-linearly separable data by transforming the original feature space into a higher-dimensional space using kernel functions, which enables the construction of an optimal separating hyperplane.

In our analysis, we compare the performance of a feedforward neural network with an SVM model for classifying handwritten digits from the MNIST dataset.

### 5. **Decision Trees**
A decision tree is a flowchart-like structure used for classification and regression tasks, where each internal node represents a decision based on an attribute, and each leaf node corresponds to a class label or a predicted value. Decision trees are constructed by recursively splitting the data into subsets based on the values of the input features, with the goal of maximizing the homogeneity of the resulting subsets. The most common splitting criteria are the Gini impurity and the information gain.

## III. Algorithm Implementation and Development
In this section, we provide an overview of the code and steps taken to fit a feedforward neural network on a dataset, compare the performance with the previous homework, and implement custom activation functions. <br>

**Creating the Neural Network Model and Preprocessing the Data** <br>
We start by importing the necessary libraries, creating a dataset, and initializing a neural network model with three layers, including two custom activation functions (sine and cosine).

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Create the dataset
X = np.arange(0, 31)
Y = np.array([...])

# Create the neural network model
model = Sequential([
    Dense(10, input_dim=1, activation='softsign'),
    Dense(20, activation='sigmoid'),
    Dense(1, activation='linear')
])
```
**Training the Neural Network Model and Computing Mean Squared Error** <br>
Next, we compile the model with a custom learning rate using the Adam optimizer, and fit the model on the dataset. We then split the dataset into training and test sets and compute the mean squared error for both.


```python
# Instantiate the Adam optimizer with a custom learning rate
custom_learning_rate = 0.02
optimizer = Adam(learning_rate=custom_learning_rate)

# Compile the model with the custom optimizer
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
model.fit(X, Y, epochs=500, verbose=0)

# Split the data into training and test sets
X_train = X[:20]
Y_train = Y[:20]
X_test = X[20:]
Y_test = Y[20:]

# Train the model on the training data
model.fit(X_train, Y_train, epochs=1000, verbose=0)

# Compute the mean squared error for the training data
Y_train_pred = model.predict(X_train)
mse_train = mean_squared_error(Y_train, Y_train_pred)

# Compute the mean squared error for the test data
Y_test_pred = model.predict(X_test)
mse_test = mean_squared_error(Y_test, Y_test_pred)

print(f"Mean squared error for training data: {mse_train}")
print(f"Mean squared error for test data: {mse_test}")
```
which yields: 

```
1/1 [==============================] - 0s 19ms/step
1/1 [==============================] - 0s 19ms/step
Mean squared error for new training data: 2.6599068913523296
Mean squared error for new test data: 13.928569659637287
```

This result is slightly better than our original solution using scipy optimize, but not by any significant margin. <br>

**Implementing Custom Activation Functions and Fitting the Model** <br>
We create custom sine and cosine activation functions and add them to the neural network model. We then compile the model with a custom learning rate using the Adam optimizer, and fit the model on the dataset.

```python
# Create a sine/cos activation function
sine_activation = tf.keras.layers.Activation(lambda x: tf.math.sin(x) + x)
cos_activation = tf.keras.layers.Activation(lambda x: tf.math.cos(x) + x)

# Create a neural network model with the custom sine activation function
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, input_dim=1, activation='linear'),
    tf.keras.layers.Dense(10),
    sine_activation,
    tf.keras.layers.Dense(10),
    cos_activation, 
    tf.keras.layers.Dense(1, activation='relu')
])

# Instantiate the Adam optimizer with a custom learning rate
custom_learning_rate = 0.005
optimizer = Adam(learning_rate=custom_learning_rate)

# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model on the new training data
model.fit(X_train_new, Y_train_new, epochs=1000, verbose=0)
```

**
Visualizing the Fitted Model and Comparing the Performance** <br>
After training the model with custom activation functions, we calculate the fitted values using the neural network and visualize the original data points along with the fitted curve.

```python
# Calculate the fitted values using the neural network
fitted_Y = model.predict(X)

# Plot the original data points
plt.scatter(X, Y, label='Data points', color='blue')

# Plot the fitted curve
plt.plot(X, fitted_Y, label='Fitted curve', color='red', linewidth=2)

# Customize the plot
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Fitting the model with a 3-layer feedforward neural network')

# Display the plot
plt.show()
```
In the comparison, the model fit using the three-layer feedforward neural network has a slightly smaller mean squared error than our data from homework 1, and looks much more generalizable, closer to a simple line through the data. The issues our very basic models from homework 1 suffered from was primarily overfitting, but with this method, it seems we may suffer from underfitting.
