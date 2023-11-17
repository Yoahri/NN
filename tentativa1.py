# PCS3438 - Inteligência Artificial - 2023/2
# Template para aula de laboratório em Redes Neurais - 20/09/2023

import numpy as np


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray):
    return x * (1 - x)


def mse_loss(y: np.ndarray, y_hat: np.ndarray):
    return np.mean(np.power(y - y_hat, 2))


def mse_loss_derivative(y: np.ndarray, y_hat: np.ndarray):
    return y_hat - y


class Layer:
    def __init__(self, input_dim: int, output_dim: int):
        self.weights = 2 * np.random.random((input_dim, output_dim)) - 1
        self.biases = np.zeros((1, output_dim))
        self.input: np.ndarray | None = None
        self.output: np.ndarray | None = None

    def forward(self, input_data) -> np.ndarray:
        """
        TODO: Forward pass of a single MLP layer

        Args:
            input_data (np.ndarray): Input data

        Returns:
            np.ndarray: Output of the layer
        """

        self.input = input_data
        self.output = sigmoid(np.dot(input_data, self.weights) + self.biases)

        return self.output

    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        TODO: Implement backward pass of a single MLP layer

        This method calculates the error of the layer and updates the weights and biases

        Args:
            output_error (np.ndarray): Error of the output layer
            learning_rate (float): Learning rate

        Returns:
            np.ndarray: Error of the previous layer
        """

        # Calcule o gradiente da camada
        delta = output_error * sigmoid_derivative(self.output)

        # Calcule o erro da camada anterior
        layer_error = delta.dot(self.weights.T)

        # Atualize os pesos e biases
        self.weights -= self.input.T.dot(delta) * learning_rate
        self.biases -= np.sum(delta, axis=0, keepdims=True) * learning_rate

        return layer_error


def forward(input: np.ndarray, layers: list[Layer]):
    """
    TODO: Applies forward pass to the MLP model

    Args:
        input (np.ndarray): Input data
        layers (list[Layer]): List of layers

    Returns:
        np.ndarray: Output of the MLP model
    """

    current_input = input
    for layer in layers:
        current_input = layer.forward(current_input)
    return current_input


def backward(
    y: np.ndarray, y_hat: np.ndarray, layers: list[Layer], learning_rate: float
) -> None:
    """
    TODO: Applies backpropagation to the MLP model

    Args:
        y (np.ndarray): Ground truth
        y_hat (np.ndarray): Predicted values
        layers (list[Layer]): List of layers
        learning_rate (float): Learning rate
    """

    output_error = mse_loss_derivative(y, y_hat)
    i = 0

    for layer in reversed(layers):
        learning_rate /=  (1 + i)
        output_error = layer.backward(output_error, learning_rate)
        i+=1


def main():
    # XOR input and output
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Hyperparameters
    hidden_layers = [2, 2]  # Two hidden layers, 2 neurons each
    epochs = 100000
    learning_rate = 0.1

    # Initialize layers
    layers = [Layer(x.shape[1], hidden_layers[0])]
    for i in range(len(hidden_layers) - 1):
        layers.append(Layer(hidden_layers[i], hidden_layers[i + 1]))
    layers.append(Layer(hidden_layers[-1], y.shape[1]))

    # Train the model
    for epoch in range(epochs):
        # Forward pass
        y_hat = forward(x, layers)

        # Loss
        loss = mse_loss(y, y_hat)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} Loss: {np.mean(loss)}")

        # Backward
        backward(y, y_hat, layers, learning_rate)

    # Test the model
    y_hat = forward(x, layers)

    print("Test input:", x)
    print("Test output:", y_hat)
    print("Learning rate:", learning_rate)


if __name__ == "__main__":
    main()
