import random
import numpy as np
import matplotlib.pyplot as plt

class Adaline:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size + 1)]
        self.learning_rate = learning_rate
        self.errors = []

    def predice(self, inputs):
        net_input = sum(weight * input_value for weight, input_value in zip(self.weights[1:], inputs)) + self.weights[0]
        return 1 / (1 + np.exp(-net_input))

    def pesos(self, inputs, target):
        error = target - self.predice(inputs)
        self.weights[0] += self.learning_rate * error
        for i in range(1, len(self.weights)):
            self.weights[i] += self.learning_rate * error * inputs[i-1]

    def entrenamiento(self, training_inputs, targets, epochs):
        for _ in range(epochs):
            for inputs, target in zip(training_inputs, targets):
                self.pesos(inputs, target)
            self.errors.append(self.calculaerror(training_inputs, targets))

    def calculaerror(self, inputs, targets):
        return np.mean([(self.predice(inputs[i]) - targets[i]) ** 2 for i in range(len(targets))])

    def prueba(self, test_inputs, targets):
        correct = sum(1 for inputs, target in zip(test_inputs, targets) if round(self.predice(inputs)) == target)
        error = self.calculaerror(test_inputs, targets) / len(test_inputs)
        accuracy = correct / len(test_inputs)
        print(f"Error promedio: {error:.2f}\nPrecisión: {accuracy:.2f}")

    def error(self):
        plt.plot(range(1, len(self.errors) + 1), self.errors, marker='o')
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.grid(True)
        plt.show()

def OUTPUT():
    adaline_model = Adaline(input_size=4, learning_rate=0.05)
    inputs = [[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1], [1,0,0,0], [1,0,0,1],[1,0,1,0], [1,0,1,1], [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1]]
    desired_outputs = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    adaline_model.entrenamiento(inputs, desired_outputs, epochs=100)
    adaline_model.prueba(inputs, desired_outputs)
    adaline_model.error()
    
OUTPUT()

