import numpy as np
import matplotlib.pyplot as plt

class CompetitiveNet:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size, output_size)

    def train(self, input_data, epochs=100):
        for _ in range(epochs):
            # Forward pass
            output = np.dot(input_data, self.weights)
            winner_idx = np.argmax(output, axis=1)

            # Update weights
            for i in range(input_data.shape[0]):
                self.weights[:, winner_idx[i]] += self.learning_rate * (input_data[i] - self.weights[:, winner_idx[i]])

    def predict(self, input_data):
        output = np.dot(input_data, self.weights)
        return np.argmax(output, axis=1)

# Generate random input data
np.random.seed(0)
input_data = np.random.rand(100, 2)  # 100 samples, 2 features

# Create an instance of the CompetitiveNet
net = CompetitiveNet(input_size=2, output_size=3, learning_rate=0.1)
net.train(input_data, epochs=100)
# Generate new test data
test_data = np.random.rand(10, 2)  # 10 samples for testing

# Predict the classes
predicted_classes = net.predict(test_data)
print("Predicted classes:", predicted_classes)

# Plotting the clusters
plt.scatter(input_data[:, 0], input_data[:, 1], c=net.predict(input_data), cmap='viridis')
plt.scatter(test_data[:, 0], test_data[:, 1], c=predicted_classes, cmap='viridis', marker='x', s=100)
plt.title('Competitive Neural Network Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()
