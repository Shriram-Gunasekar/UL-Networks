import numpy as np

class CounterpropagationNetwork:
    def __init__(self, input_size, output_size, learning_rate_unsupervised=0.1, learning_rate_supervised=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate_unsupervised = learning_rate_unsupervised
        self.learning_rate_supervised = learning_rate_supervised
        self.weights_unsupervised = np.random.rand(input_size, output_size)
        self.weights_supervised = np.random.rand(output_size, output_size)

    def train(self, input_data, labels, epochs=100):
        for _ in range(epochs):
            for x, label in zip(input_data, labels):
                # Unsupervised phase (competitive layer)
                distances = np.linalg.norm(self.weights_unsupervised - x.reshape(-1, 1), axis=0)
                winner_idx = np.argmin(distances)
                self.weights_unsupervised[:, winner_idx] += self.learning_rate_unsupervised * (x - self.weights_unsupervised[:, winner_idx])

                # Supervised phase (Grossberg layer)
                self.weights_supervised[winner_idx] += self.learning_rate_supervised * (label - self.weights_supervised[winner_idx])

    def predict(self, input_data):
        winners = []
        for x in input_data:
            distances = np.linalg.norm(self.weights_unsupervised - x.reshape(-1, 1), axis=0)
            winner_idx = np.argmin(distances)
            winners.append(winner_idx)
        return winners

# Example usage
input_data = np.array([[0.2, 0.8], [0.6, 0.4], [0.5, 0.7], [0.3, 0.3]])
labels = np.array([0, 1, 1, 0])  # Class labels for input data
cpn_net = CounterpropagationNetwork(input_size=2, output_size=2)
cpn_net.train(input_data, labels, epochs=100)

# Test with new data
test_data = np.array([[0.1, 0.9], [0.7, 0.5]])
predicted_labels = cpn_net.predict(test_data)
print("Predicted labels:", predicted_labels)
