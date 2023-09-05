get_ipython().run_line_magic('reset', '')
# Import the libraries needed for the code
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Read the data
data = pd.read_excel(' # Data path is sufficient ')# Data path is sufficient

# Delineate characteristics and target variables
X = data.drop('Voltage', axis=1)
y = data['Voltage']

# Delineate data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Delineate training, validation and test sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
model = MLPRegressor(hidden_layer_sizes=(25, 25), random_state=42)

# Setup hyperparameter search space
param_grid = {
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(model, param_grid, cv=4)
grid_search.fit(X_train_scaled, y_train)

# Access to optimal models and learning rates
best_model = grid_search.best_estimator_
best_learning_rate = grid_search.best_params_['learning_rate_init']

# Retrain the model using the optimal learning rate
best_model.set_params(learning_rate_init=best_learning_rate)
best_model.fit(X_train_scaled, y_train)

# Evaluate models on test sets
y_test_pred = best_model.predict(X_test_scaled)
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)

print("Test R^2 score: ", r2_test)
print("Test RMSE: ", rmse_test)
print("Test MAE: ", mae_test)

# Evaluate the model on the training set
y_train_pred = best_model.predict(X_train_scaled)
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
mae_train = mean_absolute_error(y_train, y_train_pred)

print("Train R^2 score: ", r2_train)
print("Train RMSE: ", rmse_train)
print("Train MAE: ", mae_train)

# Predicted data vs. real data on the test set
plt.scatter(y_test, y_test_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Test Set - Actual vs Predicted')
plt.legend()
plt.show()

# Plot of predicted data vs. real data on training set
plt.scatter(y_train, y_train_pred, color='blue', label='Predicted')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', label='Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Train Set - Actual vs Predicted')
plt.legend()
plt.show()



