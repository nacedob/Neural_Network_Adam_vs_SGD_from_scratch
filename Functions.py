import numpy as np
from abc import ABC, abstractmethod


# ------------------------------------------------------------------------------
# DEFINITION OF INTERFACE FUNCTION----------------------------------------
# ------------------------------------------------------------------------------

class CostFunction(ABC):
    @abstractmethod
    def function(self, prediction, real_output):
        pass

    @abstractmethod
    def derivative_function(self, prediction, real_output):
        pass


class ActivationFunction(ABC):
    @abstractmethod
    def function(self, x):
        pass

    @abstractmethod
    def derivative_function(self, x):
        pass


# ------------------------------------------------------------------------------
# DEFINITION OF POSSIBLE COST FUNCTIONS----------------------------------------
# ------------------------------------------------------------------------------
class QuadracticError(CostFunction):
    def function(self, prediction, real_output):
        return np.sum(np.power(prediction - real_output, 2)) / len(prediction)

    def derivative_function(self, prediction, real_output):
        return prediction - real_output


# ------------------------------------------------------------------------------
# DEFINITION OF POSSIBLE ACTIVATION FUNCTIONS----------------------------------
# ------------------------------------------------------------------------------

class Sigmoid(ActivationFunction):

    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_function(self, x):
        return np.exp(-x) * np.square(self.function(x))


class Tanh(ActivationFunction):
    def function(self, x):
        return np.tanh(x)

    def derivative_function(self, x):
        return 1 - np.power(np.tanh(x), 2)

