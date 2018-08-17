import numpy as np


class LossBase(object):
    def __init__(self):
        self.name = "Loss model - not directly usable"

    def __call__(self, y, y_t):
        """Compute loss.

        y: 1D numpy array to be processed.
        y_t: 1D numpy array corresponding to the ground truth.
        """
        pass

    def grad(self, y, y_t):
        """Compute gradient.

        y: 1D numpy array to be processed.
        y_t: 1D numpy array corresponding to the ground truth.
        """
        pass


class LossSquare(LossBase):
    def __init__(self):
        self.name = "SquaredError"

    def __call__(self, y, y_t):
        return np.square(y - y_t).sum()

    def grad(self, y, y_t):
        return 2 * (y - y_t)


class LossCrossEntropy(LossBase):
    def __init__(self):
        self.name = "CrossEntropy"

    def __call__(self, y, y_t):
        loss = np.zeros(y.shape)
        loss[y > 0] = - y_t * np.log(y)
        return loss.sum()

    def grad(self, y, y_t):
        grad = np.zeros(y.shape)
        grad[y > 0] = - y_t / y
        return grad
