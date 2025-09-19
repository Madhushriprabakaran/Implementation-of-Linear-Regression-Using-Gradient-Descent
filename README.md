# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.
```
developed by:madhushri
RegisterNumber: 212224040178 
*/
import numpy as np
import matplotlib.pyplot as plt

# Load dataset (for example purposes, we use small dummy data)
# x = population (in 10,000s), y = profit (in $10,000s)
x = np.array([6.1101, 5.5277, 8.5186, 7.0032, 5.8598])
y = np.array([17.592, 9.1302, 13.662, 11.854, 6.8233])

m = len(y)  # number of training examples

# Add intercept term to x
X = np.c_[np.ones(m), x]   # shape (m, 2)

# Initialize theta
theta = np.zeros(2)

# Hypothesis function
def hypothesis(X, theta):
    return np.dot(X, theta)

# Cost function (Mean Squared Error)
def compute_cost(X, y, theta):
    predictions = hypothesis(X, theta)
    error = predictions - y
    return (1/(2*m)) * np.dot(error, error)

# Gradient Descent
def gradient_descent(X, y, theta, alpha, iterations):
    J_history = []
    for _ in range(iterations):
        predictions = hypothesis(X, theta)
        error = predictions - y
        gradient = (1/m) * np.dot(X.T, error)
        theta -= alpha * gradient
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history

# Parameters
alpha = 0.01
iterations = 1500

# Run gradient descent
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

print("Optimized Theta:", theta)
print("Final Cost:", J_history[-1])

# Predict profit for a city with population = 7 (70,000)
population = 7.0
predicted_profit = hypothesis([1, population], theta)
print(f"Predicted Profit for population {population*10000}: {predicted_profit*10000:.2f}")

# Plot cost function convergence
plt.plot(J_history)
plt.xlabel("Iterations")
plt.ylabel("Cost J")
plt.title("Convergence of Gradient Descent")
plt.show()

# Plot regression line
plt.scatter(x, y, color="red", marker="x", label="Training Data")
plt.plot(x, hypothesis(X, theta), label="Linear Regression")
plt.xlabel("Population (10,000s)")
plt.ylabel("Profit ($10,000s)")
plt.legend()
plt.show()

```

## Output:
<img width="621" height="94" alt="image" src="https://github.com/user-attachments/assets/055d9017-3b9f-458b-a25e-4e4e190b89ef" />
<img width="1025" height="667" alt="image" src="https://github.com/user-attachments/assets/e9816702-7755-4830-acbd-beea4c18054e" />
<img width="1012" height="691" alt="image" src="https://github.com/user-attachments/assets/078376ef-8f68-4e20-a99b-20a93ecec387" />





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
