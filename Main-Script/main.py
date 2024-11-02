import sys
import os
import numpy as np

# Adding the parent directory of 'Linear_Regression_Module' to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AntColony_PyCode.ant_colony_optimization import AntColonyOptimization
from pvt_Data.pvt_data import pvt_data

def predict_gor_with_aco(pvt_data, bubble_point_pressure, api_gravity, gas_gravity, reservoir_temperature):
    # Adjust ACO parameters for better tuning
    num_ants = 20  # Increase number of ants for better exploration
    num_iterations = 200  # More iterations for convergence
    decay = 0.95
    alpha = 1.0
    beta = 2.0

    # Implement a new ACO instance and run
    aco = AntColonyOptimization(pvt_data, num_ants=num_ants, num_iterations=num_iterations, decay=decay, alpha=alpha, beta=beta)
    shortest_path, _ = aco.run()

    # Extract GOR values corresponding to the optimized parameters from the shortest path found
    optimized_gor_values = [pvt_data[i]['actual_gor'] for i in shortest_path]

    # Calculate a weighted predicted GOR based on the distance from the input parameters
    weights = []
    for i in shortest_path:
        distance = (
            abs(bubble_point_pressure - pvt_data[i]['bubble_point_pressure']) +
            abs(api_gravity - pvt_data[i]['api_gravity']) +
            abs(gas_gravity - pvt_data[i]['gas_gravity']) +
            abs(reservoir_temperature - pvt_data[i]['reservoir_temperature'])
        )
        weights.append(1 / (1 + distance))  # Weight inversely proportional to distance

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Calculate the weighted average of the GOR values
    predicted_gor = sum(g * w for g, w in zip(optimized_gor_values, normalized_weights))

    return predicted_gor

def main():
    # Example usage for new input values
    print("\nEnter bubble point pressure:")
    bubble_point_pressure = float(input())
    print("Enter API gravity:")
    api_gravity = float(input())
    print("Enter gas gravity:")
    gas_gravity = float(input())
    print("Enter reservoir temperature:")
    reservoir_temperature = float(input())

    # Predicting the GOR for new input values using ACO
    estimated_gor = predict_gor_with_aco(pvt_data, bubble_point_pressure, api_gravity, gas_gravity, reservoir_temperature)
    
    if estimated_gor is not None:
        print(f"Estimated GOR using ACO: {estimated_gor:.2f}")

if __name__ == "__main__":
    main()
