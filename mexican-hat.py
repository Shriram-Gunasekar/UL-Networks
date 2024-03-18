import numpy as np

class SelfOrganizingMap:
    def __init__(self, input_size, output_size, learning_rate=0.1, sigma=1.0):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.weights = np.random.rand(input_size, output_size)

    def train(self, input_data, epochs=100):
        for _ in range(epochs):
            for x in input_data:
                # Find the winning neuron (closest weight vector)
                distances = np.linalg.norm(self.weights - x.reshape(-1, 1), axis=0)
                winner_idx = np.argmin(distances)

                # Update weights of the winner and its neighbors
                for j in range(self.output_size):
                    influence = self._neighborhood_function(j, winner_idx)
                    self.weights[:, j] += influence * self.learning_rate * (x - self.weights[:, j])

    def _neighborhood_function(self, j, winner_idx):
        distance = abs(j - winner_idx)
        if distance <= self.sigma:
            return np.exp(-distance**2 / (2 * self.sigma**2))
        else:
            return 0.0

    def predict(self, input_data):
        winners = []
        for x in input_data:
            distances = np.linalg.norm(self.weights - x.reshape(-1, 1), axis=0)
            winner_idx = np.argmin(distances)
            winners.append(winner_idx)
        return winners

# Example usage
input_data = np.random.rand(100, 2)  # Sample input data
som = SelfOrganizingMap(input_size=2, output_size=5)
som.train(input_data, epochs=100)

# Predict clusters for new data
test_data = np.random.rand(10, 2)
predicted_clusters = som.predict(test_data)
print("Predicted clusters:", predicted_clusters)
