import numpy as np
import random
import math

# Ant Colony Optimization Algorithm class
class AntColonyOptimization:
    def __init__(self, pvt_data, num_ants=20, num_iterations=100, decay=0.95, alpha=1.0, beta=2.0):
        self.pvt_data = pvt_data  # PVT data for the optimization
        self.num_ants = num_ants  # Number of ants to simulate
        self.num_iterations = num_iterations  # Number of iterations to run the algorithm
        self.decay = decay  # Decay factor for pheromone evaporation
        self.alpha = alpha  # Importance of pheromone
        self.beta = beta  # Importance of distance (heuristic)
        self.num_nodes = len(pvt_data)  # Number of nodes (data points) to optimize

        # Initialize pheromones on all edges
        self.pheromone = np.ones((self.num_nodes, self.num_nodes))  # Initially, equal pheromone on all paths
        self.distances = self.calculate_distances()  # Distance matrix (to be used as heuristics)

    def calculate_correlations(self):
        """Calculate the correlation of each parameter with actual GOR."""
        bubble_point_pressure_values = np.array([entry['bubble_point_pressure'] for entry in self.pvt_data])
        api_gravity_values = np.array([entry['api_gravity'] for entry in self.pvt_data])
        gas_gravity_values = np.array([entry['gas_gravity'] for entry in self.pvt_data])
        reservoir_temperature_values = np.array([entry['reservoir_temperature'] for entry in self.pvt_data])
        gor_values = np.array([entry['actual_gor'] for entry in self.pvt_data])
        
        return {
            'bubble_point_pressure': np.corrcoef(bubble_point_pressure_values, gor_values)[0, 1],
            'api_gravity': np.corrcoef(api_gravity_values, gor_values)[0, 1],
            'gas_gravity': np.corrcoef(gas_gravity_values, gor_values)[0, 1],
            'reservoir_temperature': np.corrcoef(reservoir_temperature_values, gor_values)[0, 1]
        }

    def calculate_distances(self):
        """Calculate the distance between all nodes (PVT data points) based on weighted correlations."""
        correlations = self.calculate_correlations()
        distances = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    distance = (
                        abs(self.pvt_data[i]['bubble_point_pressure'] - self.pvt_data[j]['bubble_point_pressure']) * correlations['bubble_point_pressure'] +
                        abs(self.pvt_data[i]['api_gravity'] - self.pvt_data[j]['api_gravity']) * correlations['api_gravity'] +
                        abs(self.pvt_data[i]['gas_gravity'] - self.pvt_data[j]['gas_gravity']) * correlations['gas_gravity'] +
                        abs(self.pvt_data[i]['reservoir_temperature'] - self.pvt_data[j]['reservoir_temperature']) * correlations['reservoir_temperature']
                    )
                    distances[i][j] = distance
        return distances

    def run(self):
        """Run the ACO algorithm to find the best path (optimized GOR prediction)."""
        best_path = None
        best_cost = float('inf')
        
        for _ in range(self.num_iterations):
            all_paths = []
            all_costs = []
            
            for ant in range(self.num_ants):
                path = self.construct_path()
                cost = self.evaluate_path(path)
                all_paths.append(path)
                all_costs.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            self.update_pheromones(all_paths, all_costs)
        
        return best_path, best_cost

    def construct_path(self):
        """Construct a path by simulating the movement of ants."""
        path = []
        visited = set()
        current_node = random.randint(0, self.num_nodes - 1)  # Start from a random node
        path.append(current_node)
        visited.add(current_node)
        
        while len(path) < self.num_nodes:
            next_node = self.select_next_node(current_node, visited)
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
        
        return path

    def select_next_node(self, current_node, visited):
        """Select the next node to visit based on pheromone levels and distances (heuristics)."""
        probabilities = []
        for i in range(self.num_nodes):
            if i not in visited:
                pheromone_level = self.pheromone[current_node][i] ** self.alpha
                heuristic_value = (1 / (1 + self.distances[current_node][i])) ** self.beta
                probability = pheromone_level * heuristic_value
                probabilities.append(probability)
            else:
                probabilities.append(0)

        total_probability = sum(probabilities)
        probabilities = [prob / total_probability for prob in probabilities]
        
        return np.random.choice(range(self.num_nodes), p=probabilities)

    def evaluate_path(self, path):
        """Evaluate the cost of the path (using weighted GOR values and distance)."""
        optimized_gor_values = [self.pvt_data[i]['actual_gor'] for i in path]
        distance = sum(self.distances[path[i]][path[i + 1]] for i in range(len(path) - 1))
        return distance

    def update_pheromones(self, all_paths, all_costs):
        """Update pheromones based on the paths explored by ants."""
        pheromone_delta = np.zeros_like(self.pheromone)
        
        for path, cost in zip(all_paths, all_costs):
            for i in range(len(path) - 1):
                pheromone_delta[path[i]][path[i + 1]] += 1 / cost

        self.pheromone = (1 - self.decay) * self.pheromone + pheromone_delta

# Function to predict GOR using ACO
def predict_gor_with_aco(pvt_data, bubble_point_pressure, api_gravity, gas_gravity, reservoir_temperature):
    num_ants = 50
    num_iterations = 500
    decay = 0.9
    alpha = 2.0
    beta = 1.0

    aco = AntColonyOptimization(pvt_data, num_ants=num_ants, num_iterations=num_iterations, decay=decay, alpha=alpha, beta=beta)
    shortest_path, _ = aco.run()

    optimized_gor_values = [pvt_data[i]['actual_gor'] for i in shortest_path]

    weights = []
    for i in shortest_path:
        distance = (
            abs(bubble_point_pressure - pvt_data[i]['bubble_point_pressure']) +
            abs(api_gravity - pvt_data[i]['api_gravity']) +
            abs(gas_gravity - pvt_data[i]['gas_gravity']) +
            abs(reservoir_temperature - pvt_data[i]['reservoir_temperature'])
        )
        weights.append(1 / (1 + distance))

    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    predicted_gor = sum(g * w for g, w in zip(optimized_gor_values, normalized_weights))

    return predicted_gor

# Main function to take user input and predict GOR
def main():
    print("\nEnter bubble point pressure:")
    bubble_point_pressure = float(input())
    print("Enter API gravity:")
    api_gravity = float(input())
    print("Enter gas gravity:")
    gas_gravity = float(input())
    print("Enter reservoir temperature:")
    reservoir_temperature = float(input())

    pvt_data = [
        {'bubble_point_pressure': 1200, 'api_gravity': 35.0, 'gas_gravity': 0.65, 'reservoir_temperature': 150, 'actual_gor': 800},
        {'bubble_point_pressure': 1300, 'api_gravity': 30.0, 'gas_gravity': 0.70, 'reservoir_temperature': 160, 'actual_gor': 900},
        # Add more data points...
    ]
    
    estimated_gor = predict_gor_with_aco(pvt_data, bubble_point_pressure, api_gravity, gas_gravity, reservoir_temperature)
    
    if estimated_gor is not None:
        print(f"Estimated GOR using ACO with correlation weighting: {estimated_gor:.2f}")

if __name__ == "__main__":
    main()
