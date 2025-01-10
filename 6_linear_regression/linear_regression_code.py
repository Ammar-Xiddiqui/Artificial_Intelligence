import numpy as np

# Calculate the mean of values
def calculate_mean(values):
    return np.mean(values)

# Calculate the slope (theta_1)
def calculate_slope(X, Y, mean_X, mean_Y):
    numerator = np.sum((X - mean_X) * (Y - mean_Y))
    denominator = np.sum((X - mean_X) ** 2)
    return numerator / denominator

# Calculate the intercept (theta_0)
def calculate_intercept(mean_X, mean_Y, slope):
    return mean_Y - slope * mean_X

# Make predictions
def predict(X, theta_0, theta_1):
    return theta_0 + theta_1 * X

# Calculate Mean Squared Error (MSE)
def calculate_mse(Y, Y_pred):
    return np.mean((Y - Y_pred) ** 2)

# Implement gradient descent for weight adjustment
def gradient_descent(X, Y, theta_0, theta_1, learning_rate, iterations):
    m = len(Y)  # Number of data points
    for _ in range(iterations):
        Y_pred = predict(X, theta_0, theta_1)
        d_theta_0 = -2 / m * np.sum(Y - Y_pred)  # Partial derivative w.r.t theta_0
        d_theta_1 = -2 / m * np.sum((Y - Y_pred) * X)  # Partial derivative w.r.t theta_1
        theta_0 -= learning_rate * d_theta_0  # Update theta_0
        theta_1 -= learning_rate * d_theta_1  # Update theta_1
    return theta_0, theta_1

# Fit the linear regression model
def fit_linear_regression(X, Y, learning_rate=0.01, iterations=1000):
    mean_X = calculate_mean(X)
    mean_Y = calculate_mean(Y)
    initial_slope = calculate_slope(X, Y, mean_X, mean_Y)
    initial_intercept = calculate_intercept(mean_X, mean_Y, initial_slope)
    theta_0, theta_1 = gradient_descent(X, Y, initial_intercept, initial_slope, learning_rate, iterations)
    return theta_0, theta_1

# Test the model
def test_model():
    # Dataset
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    Y = np.array([30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000])

    # Fit the model
    learning_rate = 0.01
    iterations = 1000
    theta_0, theta_1 = fit_linear_regression(X, Y, learning_rate, iterations)

    # Make predictions
    Y_pred = predict(X, theta_0, theta_1)

    # Evaluate the model
    mse = calculate_mse(Y, Y_pred)

    # Print results
    print(f"Slope (theta_1): {theta_1}")
    print(f"Intercept (theta_0): {theta_0}")
    print(f"Predictions: {Y_pred}")
    print(f"Mean Squared Error (MSE): {mse}")

# Run the test
test_model()
