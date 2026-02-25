SOLUTION = """
# LINEAR REGRESSION FROM SCRATCH USING NUMPY

import numpy as np
import matplotlib.pyplot as plt


# 1. Generate Sample Data

np.random.seed(42)
N = 100

X = 10 * np.random.rand(N)

true_m = 3.0
true_b = 5.0
noise = np.random.randn(N) * 2.5
Y = true_m * X + true_b + noise


# 2. Initialize Parameters and Hyperparameters

m = 0.0
b = 0.0

learning_rate = 0.01
epochs = 1000

loss_history = []

print("Starting Gradient Descent...\\n")


# 3. Training Loop (Gradient Descent)

for epoch in range(epochs):
    # Forward pass
    Y_pred = m * X + b

    # Error
    error = Y - Y_pred

    # Loss (MSE)
    mse = (1/N) * np.sum(error**2)
    loss_history.append(mse)

    # Gradients
    dm = -(2/N) * np.sum(X * error)
    db = -(2/N) * np.sum(error)

    # Parameter update
    m = m - learning_rate * dm
    b = b - learning_rate * db

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:4d} | MSE Loss: {mse:.4f} | m: {m:.4f}, b: {b:.4f}")

print("\\nTraining Complete.")
print(f"Target parameters : m = {true_m}, b = {true_b}")
print(f"Learned parameters: m = {m:.4f}, b = {b:.4f}\\n")


# 4. Visualization

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, Y, color='blue', alpha=0.6, label='Data Points')
plt.plot(X, m * X + b, color='red', linewidth=2, label=f'Best Fit: y={m:.2f}x+{b:.2f}')
plt.title('Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(epochs), loss_history, color='green', linewidth=2)
plt.title('Mean Squared Error Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.tight_layout()
plt.show()
""".strip()

DESCRIPTION = "Implement linear regression from scratch in NumPy using gradient descent, then plot the fit and loss curve."
