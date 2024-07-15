import numpy as np

class AntColonyOptimization:
    def __init__(self, pvt_data, num_ants=10, num_iterations=100, decay=0.95, alpha=1, beta=2):
        self.pvt_data = pvt_data
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.distance_matrix = self.calculate_distance_matrix()
        self.pheromone_matrix = np.ones_like(self.distance_matrix) / len(pvt_data)
        self.shortest_path = None
        self.shortest_cost = np.inf

    def calculate_distance_matrix(self):
        num_points = len(self.pvt_data)
        dist_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    # Use a simple distance metric; can be adjusted based on your data
                    dist_matrix[i, j] = np.abs(self.pvt_data[i]['bubble_point_pressure'] - self.pvt_data[j]['bubble_point_pressure'])
        return dist_matrix

    def run(self):
        for _ in range(self.num_iterations):
            # Initialize ants
            ants_paths = self.generate_ant_paths()
            # Update pheromone levels
            self.update_pheromone(ants_paths)
            # Find the shortest path
            shortest_path, shortest_cost = self.get_shortest_path(ants_paths)
            # Update global shortest path
            if shortest_cost < self.shortest_cost:
                self.shortest_path = shortest_path
                self.shortest_cost = shortest_cost
        return self.shortest_path, self.shortest_cost

    def generate_ant_paths(self):
        num_points = len(self.pvt_data)
        ants_paths = []
        for _ in range(self.num_ants):
            start = np.random.randint(num_points)
            path = [start]
            visited = set([start])
            while len(visited) < num_points:
                probs = self.calculate_probabilities(path[-1], visited)
                next_point = np.random.choice(num_points, p=probs)
                path.append(next_point)
                visited.add(next_point)
            ants_paths.append(path)
        return ants_paths

    def calculate_probabilities(self, current_point, visited):
        pheromone = self.pheromone_matrix[current_point]
        dist = self.distance_matrix[current_point]
        unvisited_prob = np.where(np.isin(np.arange(len(pheromone)), list(visited)), 0, 1)
        row = pheromone ** self.alpha * (unvisited_prob * (1.0 / (dist + 1e-10)) ** self.beta)
        probabilities = row / np.sum(row)
        return probabilities

    def update_pheromone(self, ants_paths):
        self.pheromone_matrix *= self.decay
        for path in ants_paths:
            for i in range(len(path) - 1):
                self.pheromone_matrix[path[i], path[i+1]] += 1.0 / self.distance_matrix[path[i], path[i+1]]

    def get_shortest_path(self, ants_paths):
        shortest_cost = np.inf
        shortest_path = None
        for path in ants_paths:
            path_cost = self.calculate_path_cost(path)
            if path_cost < shortest_cost:
                shortest_cost = path_cost
                shortest_path = path
        return shortest_path, shortest_cost

    def calculate_path_cost(self, path):
        path_cost = 0
        for i in range(len(path) - 1):
            path_cost += self.distance_matrix[path[i], path[i+1]]
        return path_cost
