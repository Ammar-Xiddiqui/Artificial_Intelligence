import numpy as np
import matplotlib.pyplot as plt

# Sigmoid Function
def sigmoid(z):
    """
    Compute the sigmoid of z.
    """
    return 1 / (1 + np.exp(-z))

# Binary Cross-Entropy Loss
def cross_entropy_loss(y_true, y_pred):
    """
    Compute binary cross-entropy loss.
    """
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Avoid log(0) errors
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Gradient Descent
def gradient_descent(X, y, weights, learning_rate, iterations):
    """
    Perform gradient descent to optimize weights.
    """
    m = len(y)
    for _ in range(iterations):
        predictions = sigmoid(np.dot(X, weights))
        gradients = np.dot(X.T, (predictions - y)) / m
        weights -= learning_rate * gradients
    return weights

# Prediction
def predict(X, weights):
    """
    Predict using sigmoid function.
    """
    return sigmoid(np.dot(X, weights)) >= 0.5

# Logistic Regression Model Training
def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    """
    Fit logistic regression model.
    """
    X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
    weights = np.zeros(X.shape[1])  # Initialize weights
    weights = gradient_descent(X, y, weights, learning_rate, iterations)
    return weights

# Evaluate Model Accuracy
def evaluate(y_true, y_pred):
    """
    Evaluate accuracy of predictions.
    """
    return np.mean(y_true == y_pred)

# Plot Decision Boundary
def plot_decision_boundary(X, y, weights):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), 
                           np.linspace(x2_min, x2_max, 100))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    grid = np.c_[np.ones(grid.shape[0]), grid]  # Add bias term
    probs = sigmoid(np.dot(grid, weights)).reshape(xx1.shape)
    plt.contourf(xx1, xx2, probs, levels=[0, 0.5, 1], cmap='bwr', alpha=0.2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

# Main Program
if __name__ == "__main__":
    # Dataset
    X = np.array([[0.1, 1.1], [1.2, 0.9], [1.5, 1.6], [2.0, 1.8], [2.5, 2.1],
                  [0.5, 1.5], [1.8, 2.3], [0.2, 0.7], [1.9, 1.4], [0.8, 0.6]])
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])

    # Standardize features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_standardized = (X - X_mean) / X_std

    # Visualize the dataset
    plt.scatter(X_standardized[:, 0], X_standardized[:, 1], c=y, cmap='bwr')
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    plt.title('Data Distribution')
    plt.show()

    # Train logistic regression model
    learning_rate = 0.1
    iterations = 1000
    weights = logistic_regression(X_standardized, y, learning_rate, iterations)

    # Predict and evaluate
    X_with_bias = np.c_[np.ones(X_standardized.shape[0]), X_standardized]  # Add bias term
    y_pred = predict(X_with_bias, weights)
    accuracy = evaluate(y, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Compute cross-entropy loss
    y_prob = sigmoid(np.dot(X_with_bias, weights))
    loss = cross_entropy_loss(y, y_prob)
    print(f'Cross-Entropy Loss: {loss:.4f}')

    # Visualize the decision boundary
    plot_decision_boundary(X_standardized, y, weights)
