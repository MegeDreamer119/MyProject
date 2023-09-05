get_ipython().run_line_magic('reset', '')
# Import the libraries needed for the code
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Read the data
data = pd.read_excel(' # Data path is sufficient ')# Data path is sufficient

# Delineate data sets
X = data.drop('Voltage', axis=1)
y = data['Voltage']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the model and parametric grid
model = Lasso()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
    'tol': [0.0001, 0.001, 0.01, 0.1],
    'max_iter': [1000, 2000, 3000],
    'selection': ['cyclic', 'random']
}

# Use cross-validation and grid search to determine optimal model parameters
grid_search = GridSearchCV(model, param_grid, cv=4, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# Use optimal parameters for model training
model = Lasso(**best_params)
model.fit(X_train, y_train)

# Evaluating models on test sets  
y_test_pred = model.predict(X_test)  
r2_test = r2_score(y_test, y_test_pred)  
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))  
mae_test = mean_absolute_error(y_test, y_test_pred)  

print("Test - R²:", r2_test)  
print("Test - RMSE:", rmse_test)  
print("Test - MAE:", mae_test)

# Predicted data vs. real data on the test set
plt.scatter(y_test, y_test_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Test Set - Actual vs Predicted')
plt.legend()
plt.show()

# Plot of predicted data vs. real data on training set
y_train_pred = model.predict(X_train)
plt.scatter(y_train, y_train_pred, color='blue', label='Predicted')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', label='Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Training Set - Actual vs Predicted')  
plt.legend()
plt.show()





