import numpy as np
import matplotlib.pyplot as plt

# Perceptron Implementation
def perceptron(X, y, learning_rate=0.1, epochs=1000):
    """
    Train a Perceptron model.
    Args:
        X: Feature matrix (N x D).
        y: Labels (N x 1).
        learning_rate: Learning rate for weight updates.
        epochs: Number of iterations over the dataset.
    Returns:
        weights: Trained weights.
        bias: Trained bias.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(epochs):
        for idx, x_i in enumerate(X):
            linear_output = np.dot(x_i, weights) + bias
            y_pred = np.where(linear_output >= 0, 1, 0)  # Activation function
            update = learning_rate * (y[idx] - y_pred)
            weights += update * x_i
            bias += update

    return weights, bias

def plot_perceptron_boundary(X, y, weights, bias):
    """
    Visualize the decision boundary for the Perceptron.
    Args:
        X: Feature matrix.
        y: Labels.
        weights: Trained weights.
        bias: Trained bias.
    """
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min = -(weights[0] * x_min + bias) / weights[1]
    y_max = -(weights[0] * x_max + bias) / weights[1]
    plt.plot([x_min, x_max], [y_min, y_max], 'r-', linewidth=2)
    plt.title("Perceptron Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# XOR Neural Network Implementation
from sklearn.neural_network import MLPClassifier

def train_xor_nn(X, y):
    """
    Train a neural network to solve the XOR problem.
    Args:
        X: Feature matrix (N x D).
        y: Labels (N x 1).
    Returns:
        model: Trained MLPClassifier model.
    """
    model = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', solver='adam', max_iter=5000)
    model.fit(X, y)
    return model

def visualize_xor_boundary(model, X, y):
    """
    Visualize the decision boundary for the XOR problem.
    Args:
        model: Trained neural network model.
        X: Feature matrix.
        y: Labels.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
    plt.title("XOR Neural Network Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Example Datasets
# Perceptron dataset: Linearly separable
X_perceptron = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y_perceptron = np.array([0, 0, 1, 1])

# XOR dataset: Not linearly separable
X_xor = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y_xor = np.array([0, 0, 1, 1])

# Train and visualize Perceptron
weights, bias = perceptron(X_perceptron, y_perceptron)
plot_perceptron_boundary(X_perceptron, y_perceptron, weights, bias)

# Train and visualize XOR Neural Network
xor_model = train_xor_nn(X_xor, y_xor)
visualize_xor_boundary(xor_model, X_xor, y_xor)
