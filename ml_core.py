import numpy as np
import json

class CustomSignLanguageNN:
    def __init__(self, input_size, h1=128, h2=64, h3=32, output_size=43):
        # 43 класса по умолчанию: 33 буквы + 10 цифр
        self.sizes = [input_size, h1, h2, h3, output_size]
        
        # Инициализация весов и смещений
        self.W1 = np.random.randn(input_size, h1) * 0.01
        self.b1 = np.zeros((1, h1))
        self.W2 = np.random.randn(h1, h2) * 0.01
        self.b2 = np.zeros((1, h2))
        self.W3 = np.random.randn(h2, h3) * 0.01
        self.b3 = np.zeros((1, h3))
        self.W4 = np.random.randn(h3, output_size) * 0.01
        self.b4 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)
        
    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.relu(self.Z3)
        
        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.A4 = self.softmax(self.Z4)
        
        return self.A4

    def backward(self, X, Y, learning_rate=0.01):
        m = X.shape[0]
        
        # Градиенты выходного слоя
        dZ4 = self.A4 - Y
        dW4 = (1 / m) * np.dot(self.A3.T, dZ4)
        db4 = (1 / m) * np.sum(dZ4, axis=0, keepdims=True)
        
        # 3-й скрытый слой
        dZ3 = np.dot(dZ4, self.W4.T) * self.relu_derivative(self.Z3)
        dW3 = (1 / m) * np.dot(self.A2.T, dZ3)
        db3 = (1 / m) * np.sum(dZ3, axis=0, keepdims=True)
        
        # 2-й скрытый слой
        dZ2 = np.dot(dZ3, self.W3.T) * self.relu_derivative(self.Z2)
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # 1-й скрытый слой
        dZ1 = np.dot(dZ2, self.W2.T) * self.relu_derivative(self.Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Обновление весов
        self.W4 -= learning_rate * dW4
        self.b4 -= learning_rate * db4
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1), np.max(probabilities, axis=1)