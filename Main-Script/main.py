import sys
import os

# Adding the parent directory of 'Linear_Regression_Module' to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AntColony_PyCode.ant_colony_optimization import AntColonyOptimization
from Linear_Regression_Module.linear_regression_model import fit_linear_regression_model, predict_gor
# getting PVT data
from pvt_Data.pvt_data import pvt_data

def main():

    # Creating ACO instance and run
    aco = AntColonyOptimization(pvt_data, num_ants=10, num_iterations=100, decay=0.95, alpha=1, beta=2)
    shortest_path, shortest_cost = aco.run()

    # Extracting optimized parameters from the shortest path found
    optimized_parameters = []
    for i in shortest_path:
        optimized_parameters.append(pvt_data[i]['actual_gor'])

    print(f"Optimized parameters: {optimized_parameters}")

    # Fit a linear regression model
    model = fit_linear_regression_model(pvt_data)

    # Example usage for new input values
    print("\nEnter bubble point pressure:")
    bubble_point_pressure = float(input())
    print("Enter API gravity:")
    api_gravity = float(input())
    print("Enter gas gravity:")
    gas_gravity = float(input())
    print("Enter reservoir temperature:")
    reservoir_temperature = float(input())

    # Predicting the GOR for new input values using the fitted model
    estimated_gor = predict_gor(model, bubble_point_pressure, api_gravity, gas_gravity, reservoir_temperature)
    if estimated_gor is not None:
        print(f"Estimated GOR: {estimated_gor}")

if __name__ == "__main__":
    main()
