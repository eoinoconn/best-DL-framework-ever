import numpy as np
from  tensorflow.keras.datasets import mnist


class categorical_crossentropy():

    def __init__(self):
        pass
    
    def __call__(self, y, y_hat):
        return -np.log(y[y_hat])
    
    def derivative(self):
        pass


class softmax():
    def __init__(self):
        pass
    
    def __call__(self, inp):
        exp = np.exp(inp)
        return [x/np.sum(exp) for x in exp]

    def derivative(self):
        pass

    
class relu():
    def __init__(self):
        pass

    def __call__(self, inp):
        result = []
        for i in inp:
            if i < 0:
                result.append(0)
            else:
                result.append(i)
        return result
    
    def derivative(self, inp):
        result = []
        for i in inp:
            if i < 0:
                result.append(0)
            else:
                result.append(1)
        return np.array(result)

class Dense_layer():

    def __init__(self, neurons, activation=None):
        self.neurons = neurons
        self.activation = activation
        self.b = np.random.uniform(low=-1, high=1, size=(self.neurons))

    def __call__(self, x):
        y = np.dot(x,self.w) + self.b
        if self.activation:
            y = self.activation(y)
        return y

    def set_input_size(self, input_size):
        self.input_size = input_size
        self.w = np.random.uniform(low=-1, high=1, size=(self.input_size, self.neurons))

    def output_size(self):
        return self.neurons

    def derivative(self, z):
        if self.activation:
            z = self.activation.derivative(z)
        return z

class Model():

    def __init__(self, input_size, loss_func):
        self.model = []
        self.input_size = input_size
        self.loss_func = loss_func

    def add(self, layer):
        if len(self.model) == 0:
            layer.set_input_size(self.input_size)
            self.model.append(layer)
        else:
            layer.set_input_size(self.model[-1].output_size())
            self.model.append(layer)


    def inference(self, x):
        for layer in self.model:
            x = layer(x)
        return x

    def calculate_loss(self, x, y_hat):
        y = self.inference(x)
        return self.loss_func(y, y_hat)

    def backprop(self):
        pass



def main():
    np.random.seed(41)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    print(x_train.shape)
    
    model = Model(x_train.shape[1], categorical_crossentropy())

    model.add(Dense_layer(50, relu()))
    model.add(Dense_layer(50, relu()))
    model.add(Dense_layer(50, relu()))
    model.add(Dense_layer(50, relu()))
    model.add(Dense_layer(10, softmax()))

    h = model.calculate_loss(x_train[0], y_train[0])

    print(h)


if __name__ == "__main__":
    main()