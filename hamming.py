import numpy as np

class HammingNetwork:
    def __init__(self, input_size, memory_patterns):
        self.input_size = input_size
        self.memory_patterns = memory_patterns
        self.weights = self._train_memory_patterns()

    def _train_memory_patterns(self):
        weights = np.zeros((self.input_size, len(self.memory_patterns[0])))
        for pattern in self.memory_patterns:
            weights += np.outer(pattern, pattern)
        return weights

    def recall(self, input_pattern):
        distances = np.sum(input_pattern * self.weights, axis=0)
        recalled_pattern = self.memory_patterns[np.argmax(distances)]
        return recalled_pattern

# Example usage
memory_patterns = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])  # Sample memory patterns
hamming_net = HammingNetwork(input_size=3, memory_patterns=memory_patterns)

# Recall a pattern
input_pattern = np.array([1, 0, 0])  # Input pattern with noise
recalled_pattern = hamming_net.recall(input_pattern)
print("Recalled pattern:", recalled_pattern)
