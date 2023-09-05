 get_ipython().run_line_magic('reset', '')
# Import the libraries needed for the code
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga
# please refer to https://pypi.org/project/geneticalgorithm/1.0.2/
import seaborn as sns
import matplotlib.pyplot as plt

# Random number seeding; facilitates some random tasks where each operator gets the same result
np.random.seed(# random seed number)

# Read the data
data = pd.read_excel(' # Data path is sufficient ')# Data path is sufficient

# Delineate data sets
X = data.drop('Voltage', axis=1)
y = data['Voltage']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the model and parametric grid
model = xgb.XGBRegressor()
param_grid = {
    'max_depth': [2, 3, 4, 5, 6],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100,150, 200, 250, 300]
}

# Use cross-validation and grid search to determine optimal model parameters
grid_search = GridSearchCV(model, param_grid, cv=4, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Use optimal parameters for model training
model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

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

# Calculate the order of importance of features
feature_importances = model.feature_importances_

# Print feature importance ranking
sorted_indices = np.argsort(feature_importances)[::-1]
for i in sorted_indices:
    print(f"{X_val.columns[i]}: {feature_importances[i]}")
 
# Calculate the SHAP value
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Delete the 'Current density' feature and adjust the columns of the SHAP value matrix
X_test_without_current_density = X_test.drop('Current density', axis=1)

current_density_index = X_test.columns.get_loc('Current density')
shap_values_without_current_density = shap_values.values[:, :current_density_index]

shap_values_without_current_density = np.hstack((shap_values_without_current_density, shap_values.values[:, current_density_index + 1:]))#拼接到一起

# Mapping of Globle Feature Plot (without 'Current density')
shap.summary_plot(shap_values_without_current_density, X_test_without_current_density, feature_names=X_test_without_current_density.columns)

# Calculate the partial dependency graph of the input feature
feature_name = '# input feature' # input feature
feature_index = X_test_without_current_density.columns.get_loc(feature_name)

# Calculate partial dependency values
shap_values_feature = shap_values_without_current_density[:, feature_index]
expected_value = shap_values.base_values
feature_values = X_test_without_current_density[feature_name]

plt.figure(figsize=(10, 6))
plt.scatter(feature_values, shap_values_feature)
plt.xlabel(feature_name)
plt.ylabel('SHAP Value')
plt.title(f'Partial Dependence Plot for {feature_name}')
plt.show()

# Define the objective function
def objective_function(x):
    # Mapping the input x to the genetic algorithm to the actual range of values
    mapped_x = {
        'Heating plate temperature': 90,
        'Water content': x[0] * (0.6 - 0.05) + 0.05,
        'Ink flow rate': 1,
        'Hot pressing time': x[1] * (900 - 300) + 300,
        'Hot pressing pressure': 550,
        'Cathode compression rate': x[2] * (0.4 - 0.08) + 0.08,
        'Anode PTL porosity': 1,
        'MEA configuration': 1,
        'Titanium felt with or without platinum plating': 1,
        'Membrane thickness': 127,
        'Cathode catalyst loading': 0.13,
        'Anode catalyst loading': x[3] * (3 - 1.5) + 1.5,
        'Anode ionomer content': x[4] * (0.15 - 0.05) + 0.05,
        'Operating temperature': 80,
        'Deionized water flow rate': 20,
        'Current density': 3
    }
    
    # Use the model to predict the output value Voltage
    predicted_voltage = model.predict(pd.DataFrame(mapped_x, index=[0]))
    
    # Returns the value of Voltage
    return predicted_voltage[0]

# Generate an index of random populations
population_indices = np.random.choice(len(X_train), population_size, replace=False)

# Select correspondingly indexed samples from the training dataset as the initial population
initial_population = X_train.iloc[population_indices].values

# Define the parameters of the genetic algorithm
algorithm_param = {
    'max_num_iteration': 250,
    'population_size': 20,
    'mutation_probability': 0.1,
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None,
    'maximization_problem': True
}

# Define the genetic algorithm object
algorithm = ga(
    function=objective_function,
    dimension=5,
    variable_type='real',
    variable_boundaries=np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]),
    algorithm_parameters=algorithm_param
)

# Manually set initial populations
algorithm.population = initial_population

# Run the genetic algorithm
algorithm.run()

# Output the found optimal solution
best_solution = algorithm.best_variable
mapped_best_solution = {
    'Heating plate temperature': 90,
    'Water content': best_solution[0] * (0.6 - 0.05) + 0.05,
    'Ink flow rate': 1,
    'Hot pressing time': best_solution[1] * (900 - 300) + 300,
    'Hot pressing pressure': 550,
    'Cathode compression rate': best_solution[2] * (0.4 - 0.08) + 0.08,
    'Anode PTL porosity': 1,
    'MEA configuration': 1,
    'Titanium felt with or without platinum plating': 1,
    'Membrane thickness': 127,
    'Cathode catalyst loading': 0.13,
    'Anode catalyst loading': best_solution[3] * (3 - 1.5) + 1.5,
    'Anode ionomer content': best_solution[4] * (0.15 - 0.05) + 0.05,
    'Operating temperature': 80,
    'Deionized water flow rate': 20,
    'Current density': 3
}

best_fitness = objective_function(best_solution)

print("Best solution:", mapped_best_solution)
print("Best fitness:", best_fitness)

#Manually input fabrication parameters
input_values = {
    'Heating plate temperature': 
    'Water content': 
    'Ink flow rate': 
    'Hot pressing time': 
    'Hot pressing pressure': 
    'Cathode compression rate': 
    'Anode PTL porosity': 
    'MEA configuration': 
    'Titanium felt with or without platinum plating': 
    'Membrane thickness': 
    'Cathode catalyst loading': 
    'Anode catalyst loading': 
    'Anode ionomer content':
    'Operating temperature': 
    'Deionized water flow rate': 
    'Current density': 
}

input_df = pd.DataFrame(input_values, index=[0])

# Get output voltage
voltage = model.predict(input_df)[0]
print("Voltage:", voltage)






