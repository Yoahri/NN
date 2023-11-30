
# PCS3438 - Inteligência Artificial - 2023/2
# Template para aula de laboratório em Redes Neurais - 20/09/2023
# Vitor Peres de Brito-12717040
# Luis Eduardo Exposto Novoselecki - 11208200

import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
def confusion_matrix_custom(y_true, y_pred):
    true_positive=0
    true_negative=0
    false_positive=0
    false_negative=0
    for i in range(0, len(y_true)):
        if y_true[i]==0:
            if y_pred[i]==0:
                true_negative+=1
            else:
                false_positive+=1
        else:
            if y_true[i]==1:
                if y_pred[i]==1:
                    true_positive+=1
                else:
                    false_negative+=1

    return true_positive, false_positive, false_negative, true_negative


  


  

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mse_loss(y, y_hat):
    return np.mean(np.power(y - y_hat, 2))

def mse_loss_derivative(y, y_hat):
    return y_hat - y

def z_score_normalization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std

class Layer:
    def __init__(self, input_dim, output_dim, regularization_rate=0.01):
        self.weights = 2 * np.random.random((input_dim, output_dim)) - 1
        self.biases = np.zeros((1, output_dim))
        self.input = None
        self.output = None
        self.regularization_rate = regularization_rate

    def forward(self, input_data):
        self.input = input_data
        self.output = sigmoid(np.dot(input_data, self.weights) + self.biases)
        return self.output

    def backward(self, output_error, learning_rate):
        delta = output_error * sigmoid_derivative(self.output)
        layer_error = delta.dot(self.weights.T)

        # Regularization term
        regularization_term = self.regularization_rate * np.sum(self.weights)

        # Update weights with regularization
        self.weights -= (self.input.T.dot(delta) + regularization_term) * learning_rate
        self.biases -= np.sum(delta, axis=0, keepdims=True) * learning_rate

        return layer_error

def forward(input_data, layers):
    current_input = input_data
    for layer in layers:
        current_input = layer.forward(current_input)
    return current_input

def backward(y, y_hat, layers, learning_rate):
    output_error = mse_loss_derivative(y, y_hat)
    i = 0
    for layer in reversed(layers):
        output_error = layer.backward(output_error, learning_rate)
        i += 1

def generate_batches(x, y, batch_size):
    num_batches = len(x) // batch_size
    batches = [(x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]) for i in range(num_batches)]
    return batches

def train_test_split_custom(x, y, test_size, random_state):
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(len(x))
    np.random.shuffle(indices)

    test_size = int(test_size * len(x))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    x_train, y_train = x[train_indices], y[train_indices]
    x_test, y_test = x[test_indices], y[test_indices]

    return x_train, x_test, y_train, y_test

def kfold(x, y, array_alpha, array_lr, array_hiddenL):
    k = 5
    acuracias = []
    loss_array=[]
    fold_size = len(x) // k
    for regularization_rate in array_alpha:
        for learning_rate in array_lr:
            for hidden_layers in array_hiddenL:
                acuracia = []
                loss_aux=[]
                for i in range(k):
                    start_idx_val = i * fold_size
                    end_idx_val = (i + 1) * fold_size

                    start_idx_next_val = end_idx_val
                    end_idx_next_val = (i + 2) * fold_size if i < k - 1 else len(x)

                    x_train = np.concatenate((x[:start_idx_val], x[end_idx_next_val:]), axis=0)
                    y_train = np.concatenate((y[:start_idx_val], y[end_idx_next_val:]), axis=0)
                    x_test = x[start_idx_val:end_idx_val]
                    y_test= y[start_idx_val:end_idx_val]

  
                    layers = [Layer(x_train.shape[1], hidden_layers[0], regularization_rate)]
                    for j in range(1, len(hidden_layers)):
                        layers.append(Layer(hidden_layers[j - 1], hidden_layers[j], regularization_rate))
                    layers.append(Layer(hidden_layers[-1], 1, regularization_rate))  # Única unidade de saída

                    epochs = 1000
                    batch_size = 32

                    for epoch in range(epochs):
                        batches = generate_batches(x_train, y_train, batch_size)
                        for batch_x, batch_y in batches:
                            y_hat = forward(batch_x, layers)
                            loss = mse_loss(batch_y, y_hat) + sum(layer.regularization_rate * np.sum(layer.weights) for layer in layers)
                            backward(batch_y, y_hat, layers, learning_rate)

                    x_test_normalized = x_test 
                    y_hat_test = forward(x_test_normalized, layers)
                    test_loss = mse_loss(y_test, y_hat_test)
                    y_pred_labels = np.where(y_hat_test > 0.5, 1, 0).flatten()
                    tp, fp, fn, tn = confusion_matrix_custom(y_test.flatten(), y_pred_labels)
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    acuracia.append(accuracy)
                    loss_aux.append(np.mean(loss))
                loss_array.append(np.mean(loss_aux))
                acuracias.append({
                    'regularization_rate': regularization_rate,
                    'learning_rate': learning_rate,
                    'hidden_layers': hidden_layers,
                    'mean_accuracy': sum(acuracia) / len(acuracia)
                })
  
    return [acuracias,loss_array]

def main():
    
   
    data = load_breast_cancer()
    x = data.data
    y = data.target.reshape(-1, 1)

   
    x, mean, std = z_score_normalization(x)

   
    x_train, x_test, y_train, y_test = train_test_split_custom(x, y, test_size=0.2, random_state=4)

    # Hyperparameters
    hidden_layers = [2,2]  
    epochs = 1000
    learning_rate = 0.1
    regularization_rate = 0.1
    batch_size = 32

    # regularization term
    layers = [Layer(x_train.shape[1], hidden_layers[0], regularization_rate)]
    for i in range(len(hidden_layers) - 1):
        layers.append(Layer(hidden_layers[i], hidden_layers[i + 1], regularization_rate))
    layers.append(Layer(hidden_layers[-1], y_train.shape[1], regularization_rate))

    # Train the model with batches
    for epoch in range(epochs):
        batches = generate_batches(x_train, y_train, batch_size)
        for batch_x, batch_y in batches:
          
            y_hat = forward(batch_x, layers)

            loss = mse_loss(batch_y, y_hat) + sum(layer.regularization_rate * np.sum(layer.weights) for layer in layers)

            backward(batch_y, y_hat, layers, learning_rate)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {np.mean(loss)}")

    # Test the model on the test set
    # Normalize test set using mean and std from training set
    x_test_normalized = x_test 
    y_hat_test = forward(x_test_normalized, layers)
    test_loss = mse_loss(y_test, y_hat_test)
    print("Test input:", x_test_normalized)
    print("Test output:", y_hat_test)
    print("Learning rate:", learning_rate)

    # Convert predicted probabilities to class labels (0 or 1)
    y_pred_labels = np.where(y_hat_test > 0.5, 1, 0).flatten()

    # confusion matrix
    tp,fp, fn, tn = confusion_matrix_custom(y_test.flatten(), y_pred_labels)
    
    erro = 0
   
    print("Confusion Matrix:")
    print(f"True Positive: {tp}")
    print(f"True Negative: {tn}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")

    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"Accuracy: {accuracy}")
    
    kfold1=kfold(x,y,[0.01],[0.001,0.01,0.1],[[2,2]])
    for g in kfold1[0]:
      print(g)
    plt.xlabel("learning rate")
    plt.plot([0.001,0.01,0.1],kfold1[1])
    plt.show()
    #melhor valor de learning rate baseado em acuracia = 0.01

    kfold1=kfold(x,y,[0.001, 0.01, 0.1],[0.01],[[2,2]])
    for g in kfold1[0]:
      print(g)
    plt.xlabel("regularization term")
    plt.plot([0.001,0.01,0.1],kfold1[1])
    plt.show()
    #melhor valor de regularization term baseado em acuracia = 0.01
    
    kfold1=kfold(x,y,[0.01],[0.01],[[2,2], [4,4], [6,6], [8,8]])
    for g in kfold1[0]:
      print(g)
    #melhor valor de hidden layers baseado em acuracia = [2,2]

if __name__ == "__main__":
    main()