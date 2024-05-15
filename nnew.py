import numpy as np

class Node:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.activation_function = self.sigmoid
        self.activation_derivative = self.sigmoid_derivative
        self.inputs = None
        self.output = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.inputs = inputs
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        self.output = self.activation_function(weighted_sum)
        return self.output

    def backward(self, delta, learning_rate):
        delta_weights = delta * self.inputs
        delta_bias = delta
        delta_inputs = self.weights * delta
        self.weights -= learning_rate * delta_weights
        self.bias -= learning_rate * delta_bias
        return delta_inputs

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layers = [self.create_layer(layer_sizes[i], layer_sizes[i-1]) for i in range(1, self.num_layers)]
    
    def create_layer(self, num_nodes, num_inputs):
        return [Node(num_inputs) for _ in range(num_nodes)]

    def forward(self, inputs):
        activations = inputs
        for layer in self.layers:
            activations = np.array([node.forward(activations) for node in layer])
        return activations

    def backward(self, inputs, outputs, learning_rate):
        predicted_outputs = self.forward(inputs)

        # Calculate output layer deltas
        output_layer = self.layers[-1]
        output_deltas = (outputs - predicted_outputs) * predicted_outputs * (1 - predicted_outputs)

        # Backpropagate deltas through the network
        for i in range(self.num_layers - 2, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]

            deltas = []
            for j, node in enumerate(layer):
                delta = np.sum([node.weights[j] * next_node.backward(output_deltas[k], learning_rate)
                                for k, next_node in enumerate(next_layer)])
                deltas.append(delta)
            output_deltas = np.array(deltas)

    def train(self, inputs, outputs, epochs, learning_rate):
        for epoch in range(epochs):
            for input_, output_ in zip(inputs, outputs):
                self.backward(input_, output_, learning_rate)

            # Calculate and print mean squared error for monitoring
            predicted_outputs = np.array([self.forward(input_) for input_ in inputs])
            mse = np.mean(np.square(outputs - predicted_outputs))
            print(f"Epoch {epoch + 1}/{epochs}, Mean Squared Error: {mse}")

    def predict(self, inputs):
        return self.forward(inputs)

# Example usage
if __name__ == "__main__":
    # Specify the layer sizes
    layer_sizes = [2, 4, 1]  # Input layer has 2 nodes, hidden layer has 4 nodes, output layer has 1 node

    # Sample inputs and outputs
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    # Initialize and train the neural network
    nn = NeuralNetwork(layer_sizes)
    nn.train(inputs, outputs, epochs=1000, learning_rate=0.1)

    # Test the trained network
    test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = nn.predict(test_inputs)
    print("Predictions:")
    print(predictions)