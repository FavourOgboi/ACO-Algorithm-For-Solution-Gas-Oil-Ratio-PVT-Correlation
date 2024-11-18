import pandas as pd
import sys
import os

# Adding the parent directory of 'Linear_Regression_Module' to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AntColony_PyCode.volvo_aco_algorithm import AntColonyOptimization
from Linear_Regression_Module.linear_regression_model import fit_linear_regression_model, predict_gor

# Load the data from CSV
pvt_data_df = pd.read_csv("C:/Users/hp/Documents/GitHub/ACO-Algorithm-For-Solution-Gas-Oil-Ratio-PVT-Correlation/pvt_Data/cleaned_production_data.csv")

# Sample 50 rows from the DataFrame
sampled_data_df = pvt_data_df.sample(n=100, random_state=1)

print(sampled_data_df)

# Create ACO instance and run
aco = AntColonyOptimization(sampled_data_df, num_ants=10, num_iterations=100, decay=0.95, alpha=1, beta=2)
results = aco.run()

# Output the results
shortest_path_indices = results['shortest_path']
aco_gor_values = results['gor_values']

# Extract actual GOR values from the sampled DataFrame using shortest path indices
actual_gor_values = [sampled_data_df.iloc[i]['Calculated_GOR'] for i in shortest_path_indices]

# Print results
print(f"Shortest Path Indices: {shortest_path_indices}")
print(f"ACO GOR Values: {aco_gor_values}")
print(f"Actual GOR Values: {actual_gor_values}")

# Compare and check if they are in the same order
comparison_results = []
for aco_value, actual_value in zip(aco_gor_values, actual_gor_values):
    comparison_results.append((aco_value, actual_value, aco_value == actual_value))

# Display comparison results
for aco_value, actual_value, is_equal in comparison_results:
    equality_str = "Equal" if is_equal else "Not Equal"
    print(f"ACO GOR: {aco_value}, Actual GOR: {actual_value} - {equality_str}")

