import numpy as np

class LVQNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size, output_size)

    def train(self, input_data, labels, epochs=100):
        for _ in range(epochs):
            for x, label in zip(input_data, labels):
                winner_idx = self._find_winner(x)
                if label == winner_idx:
                    self.weights[:, winner_idx] += self.learning_rate * (x - self.weights[:, winner_idx])
                else:
                    self.weights[:, winner_idx] -= self.learning_rate * (x - self.weights[:, winner_idx])

    def _find_winner(self, x):
        distances = np.linalg.norm(self.weights - x.reshape(-1, 1), axis=0)
        return np.argmin(distances)

    def predict(self, input_data):
        winners = []
        for x in input_data:
            winner_idx = self._find_winner(x)
            winners.append(winner_idx)
        return winners

# Example usage
input_data = np.array([[0.2, 0.8], [0.6, 0.4], [0.5, 0.7], [0.3, 0.3]])
labels = np.array([0, 1, 1, 0])  # Class labels for input data
lvq_net = LVQNetwork(input_size=2, output_size=2)
lvq_net.train(input_data, labels, epochs=100)

# Test with new data
test_data = np.array([[0.1, 0.9], [0.7, 0.5]])
predicted_labels = lvq_net.predict(test_data)
print("Predicted labels:", predicted_labels)
