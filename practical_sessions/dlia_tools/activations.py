import numpy as np


class ActivationBase(object):
    def __init__(self):
        self.name = "Activation model - not directly usable"

    def __call__(self, x):
        """Compute activation

        x: 1D numpy array to be processed
        """
        pass

    def grad(self, x):
        """Compute gradient.

        x: 1D numpy array to be processed
        """
        pass


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ActivationSigmoid(ActivationBase):
    def __init__(self):
        self.name = "sigmoid"

    def __call__(self, x):
        return sigmoid(x)

    def grad(self, x):
        s = sigmoid(x)
        return s * (1 - s)


class ActivationRelu(ActivationBase):
    def __init__(self):
        self.name = "ReLU"

    def __call__(self, x):
        return np.maximum(x, 0)

    def grad(self, x):
        return 1. * (x > 0)


class ActivationTanh(ActivationBase):
    def __init__(self):
        self.name = "tanh"

    def __call__(self, x):
        return np.tanh(x)

    def grad(self, x):
        return 1 - np.tanh(x) ** 2


class ActivationIdentity(ActivationBase):
    def __init__(self):
        self.name = "identity"

    def __call__(self, x):
        return np.copy(x)

    def grad(self, x):
        return np.ones(len(x))


class ActivationBinary(ActivationBase):
    def __init__(self):
        self.name = "binary"

    def __call__(self, x):
            return 1 * (x > 0)

    def grad(self, x):
        return 1 * (x == 0)
