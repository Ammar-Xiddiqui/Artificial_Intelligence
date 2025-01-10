import numpy as np
import matplotlib.pyplot as plt

# Step 1: Dataset Setup
data = np.array([
    [0.1, 0.6, 1],
    [0.15, 0.71, 1],
    [0.25, 0.8, 1],
    [0.35, 0.45, 1],
    [0.5, 0.5, 0],
    [0.6, 0.2, 0],
    [0.65, 0.3, 0],
    [0.8, 0.35, 0]
])

X = data[:, :2]
y = data[:, 2]

# Step 2: Initialize weights and biases
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights = {
        "W1": np.random.randn(input_size, hidden_size),
        "b1": np.zeros((1, hidden_size)),
        "W2": np.random.randn(hidden_size, output_size),
        "b2": np.zeros((1, output_size))
    }
    return weights

# Step 3: Implement forward propagation
def forward_propagation(X, weights):
    Z1 = np.dot(X, weights["W1"]) + weights["b1"]
    A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation
    Z2 = np.dot(A1, weights["W2"]) + weights["b2"]
    A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

# Step 4: Compute the loss
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -(1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# Step 5: Implement backward propagation
def backward_propagation(X, y, weights, cache):
    m = X.shape[0]
    A1, A2 = cache["A1"], cache["A2"]

    dZ2 = A2 - y.reshape(-1, 1)
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = np.dot(dZ2, weights["W2"].T) * A1 * (1 - A1)
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

# Step 6: Update weights
def update_parameters(weights, gradients, learning_rate):
    weights["W1"] -= learning_rate * gradients["dW1"]
    weights["b1"] -= learning_rate * gradients["db1"]
    weights["W2"] -= learning_rate * gradients["dW2"]
    weights["b2"] -= learning_rate * gradients["db2"]
    return weights

# Step 7: Training loop
def train_network(X, y, hidden_size, learning_rate, epochs):
    input_size = X.shape[1]
    output_size = 1
    weights = initialize_parameters(input_size, hidden_size, output_size)
    losses = []

    for epoch in range(epochs):
        y_pred, cache = forward_propagation(X, weights)
        loss = compute_loss(y, y_pred)
        losses.append(loss)

        gradients = backward_propagation(X, y, weights, cache)
        weights = update_parameters(weights, gradients, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return weights, losses

# Step 8: Plot decision boundary
def plot_decision_boundary(X, y, weights):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    A2, _ = forward_propagation(grid, weights)
    predictions = A2.reshape(xx.shape) > 0.5

    plt.contourf(xx, yy, predictions, alpha=0.6, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Decision Boundary")
    plt.show()

# Training the network
hidden_size = 3
learning_rate = 0.1
epochs = 1000

trained_weights, losses = train_network(X, y, hidden_size, learning_rate, epochs)

# Plotting the decision boundary
plot_decision_boundary(X, y, trained_weights)

# Plotting the loss curve
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()
