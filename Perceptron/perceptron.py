#*********************************
# Author: Bryan Medina
# Mail: bryan.anidem@gmail.com
#*********************************

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, num_inputs, lr, epochs, pesos=None):
        if pesos:
            self.weights = pesos # en caso de contar con pesos del perceptron cargarlos
        else:
            self.weights = np.random.rand(num_inputs+1) # el peso extra es el bias
            
        self.lr = lr #learning rate
        self.epochs = epochs #num de iteraciones

    def act_fn(self, x, funcion='step'):
        """función de activación """
        if funcion == 'step':    
            return np.where(x>0, 1, 0)
        if funcion == 'sigmoid':
            return 1/(1 + np.exp(-x)) 
        if funcion == 'relu':
            return np.where(x>0, x, 0)
        if funcion == 'tanh':
            return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


    def predict(self, inputs):
        """feedforward del perceptron"""
        return self.act_fn(np.dot(inputs, self.weights[1:]) + self.weights[0])

    def train(self, in_train, clases, verbose=False):
        """entrenamiento del perceptron"""
        errores = []
        for _ in range(self.epochs):
            e = 0
            for entrada, clase in zip(in_train, clases):
                prediccion = self.predict(entrada)
                error = clase - prediccion
                self.weights[0] += self.lr * error
                self.weights[1:] += self.lr * error * entrada
            
                e += np.abs(error)   
            errores.append(e)
        
        if verbose:
            plt.figure(), plt.plot(errores), plt.title('errores'), plt.show()
                  
    def get_weights(self):
        """recupero los pesos de la red, útil para no volver a entrenar"""
        return self.weights


if __name__ == "__main__":

    #pesos de red entrenada [-2.1683108   1.96946991  0.86928306]

    perceptron = Perceptron(2, lr=1, epochs=10)
    
    # vemos las funciones de activacion
    x = np.arange(-10, 10, 0.1)
    plt.figure(), plt.plot(x, perceptron.act_fn(x, 'relu')), plt.show()
    
    # vals = np.array([[0,0], [0,1], [1,0], [1,1]])
    # clase = np.array([0, 0, 0, 1])

    # perceptron.train(vals, clase, True)

    # for val in vals:
    #     print(perceptron.predict(val))

    # print(perceptron.get_weights())
    