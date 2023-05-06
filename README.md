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
