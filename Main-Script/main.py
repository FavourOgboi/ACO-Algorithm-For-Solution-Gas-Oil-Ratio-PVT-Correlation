import sys
import os

# Add the parent directory of 'Linear_Regression_Module' to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AntColony_PyCode.ant_colony_optimization import AntColonyOptimization
from Linear_Regression_Module.linear_regression_model import fit_linear_regression_model, predict_gor


def main():
    # Example PVT data
    pvt_data = [
        {'bubble_point_pressure': 2405, 'api_gravity': 37, 'gas_gravity': 0.743, 'reservoir_temperature': 129, 'actual_gor': 737},
        {'bubble_point_pressure': 2200, 'api_gravity': 37, 'gas_gravity': 0.743, 'reservoir_temperature': 129, 'actual_gor': 684},
        {'bubble_point_pressure': 1950, 'api_gravity': 37, 'gas_gravity': 0.743, 'reservoir_temperature': 129, 'actual_gor': 620},
        {'bubble_point_pressure': 1700, 'api_gravity': 37, 'gas_gravity': 0.743, 'reservoir_temperature': 129, 'actual_gor': 555},
        {'bubble_point_pressure': 1450, 'api_gravity': 37, 'gas_gravity': 0.743, 'reservoir_temperature': 129, 'actual_gor': 492},
        {'bubble_point_pressure': 1200, 'api_gravity': 37, 'gas_gravity': 0.743, 'reservoir_temperature': 129, 'actual_gor': 429},
        {'bubble_point_pressure': 950, 'api_gravity': 37, 'gas_gravity': 0.743, 'reservoir_temperature': 129, 'actual_gor': 365},
        {'bubble_point_pressure': 700, 'api_gravity': 37, 'gas_gravity': 0.743, 'reservoir_temperature': 129, 'actual_gor': 301},
        {'bubble_point_pressure': 450, 'api_gravity': 37, 'gas_gravity': 0.743, 'reservoir_temperature': 129, 'actual_gor': 235},
        {'bubble_point_pressure': 200, 'api_gravity': 37, 'gas_gravity': 0.743, 'reservoir_temperature': 129, 'actual_gor': 155},
    ]

    # Create ACO instance and run
    aco = AntColonyOptimization(pvt_data, num_ants=10, num_iterations=100, decay=0.95, alpha=1, beta=2)
    shortest_path, shortest_cost = aco.run()

    # Extract optimized parameters from the shortest path found
    optimized_indices = shortest_path
    optimized_parameters = [pvt_data[i]['actual_gor'] for i in optimized_indices]
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

    # Predict the GOR for new input values using the fitted model
    estimated_gor = predict_gor(model, bubble_point_pressure, api_gravity, gas_gravity, reservoir_temperature)
    if estimated_gor is not None:
        print(f"Estimated GOR: {estimated_gor}")

if __name__ == "__main__":
    main()
