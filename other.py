# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# %%
# Assuming we have some data
# X represents house size (independent variable)
# Y represents house price (dependent variable)
X = np.random.rand(100, 1) * 5
Y = 3*X + np.random.randn(100, 1)

# %%
# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# %%
# Create a linear regression object
regr = LinearRegression()

# %%
# Train the model
regr.fit(X_train, Y_train)

# %%
# Make predictions using the testing set
Y_pred = regr.predict(X_test)

# %%
# The residuals
residuals = Y_test - Y_pred

# %%
# Calculate the RSS
RSS = (residuals**2).sum()
print(f'Residual Sum of Squares (RSS): {RSS}')

# %%
# Residual plot
plt.scatter(Y_pred, residuals)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='-')  # zero line
plt.show()

# %%
r2 = r2_score(Y_test, Y_pred)
print(f'R-squared: {r2}')

n = X_test.shape[0]  # number of observations
p = X_test.shape[1]  # number of predictors
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print(f'Adjusted R-squared: {adj_r2}')

# Scatter plot of observed vs. predicted values
plt.scatter(Y_test, Y_pred)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)  # line of perfect prediction
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('True vs. Predicted Values (with R-squared)')
plt.show()

# %%
# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set a random seed for reproducibility
np.random.seed(42)

# Generate some data
X = np.random.rand(100, 5) # random data with 5 predictors for simplicity
Y = 3*X[:,0] + np.random.randn(100) # Y depends only on the first column of X

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a linear regression object
regr = LinearRegression()

# We will train and test the model for an increasing number of predictors
for p in range(1, 6):
    # Train the model with p predictors
    regr.fit(X_train[:, :p], Y_train)

    # Make predictions using the testing set
    Y_pred = regr.predict(X_test[:, :p])

    # Calculate R-squared
    r2 = r2_score(Y_test, Y_pred)
    print(f'Number of predictors: {p}, R-squared: {r2:.2f}')

    # Calculate adjusted R-squared
    n = X_test.shape[0]  # number of observations
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    print(f'Number of predictors: {p}, Adjusted R-squared: {adj_r2:.2f}')
