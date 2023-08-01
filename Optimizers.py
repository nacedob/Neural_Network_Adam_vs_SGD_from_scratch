# ------------------------------------------------------------------------------
# OPTIMIZERS-------------------------------------------------------------------
# ------------------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    """
        Interface to represent learning algorithms to tune
        weights and biases. It contains an abstract class train
    """

    @abstractmethod
    def train(self, weights, derivative_weights, biases, derivative_biases, **kwargs):
        pass


class SGD(Optimizer):
    """
    Stochastic gradient descent.
    """

    def __init__(self, learning_rate):
        """
        SDG constructor. Specify learning_rate
        """
        self.learning_rate = learning_rate

        # Print to console
        print("%%%%%%%%%% SGD %%%%%%%%%%")

    def train(self, weights, derivative_weights, biases, derivative_biases, **kwargs):
        """
        Stochastic gradient descent to train a NeuralNetwork class.
        kwargs is added to admit natural arguments for Adam. Betweem those additional
        arguments, there must be one called mini_batch_size
        """

        # SGD update
        weights = [weight_layer - self.learning_rate * derivative_weight_layer / kwargs["mini_batch_size"]
                   for weight_layer, derivative_weight_layer in zip(weights, derivative_weights)]
        biases = [bias_layer - -self.learning_rate * derivative_bias_layer / kwargs["mini_batch_size"]
                  for bias_layer, derivative_bias_layer in zip(biases, derivative_biases)]

        return weights, biases


class Adam(Optimizer):

    def __init__(self, architecture, learning_rate=1e3, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        """
        Adam constructor. Specify learning_rate, beta_1, beta_2 and epsilon
        """

        # Initizalize momentums
        self.m_biases = [np.zeros([layer, 1]) for layer in architecture[1:]]
        self.v_biases = [np.zeros([layer, 1]) for layer in architecture[1:]]
        self.m_weights = [np.zeros([currentLayer, previousLayer])
                          for currentLayer, previousLayer
                          in zip(architecture[1:], architecture[:-1])]
        self.v_weights = [np.zeros([currentLayer, previousLayer])
                          for currentLayer, previousLayer
                          in zip(architecture[1:], architecture[:-1])]

        # Hiperparameters
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # Print to console
        print("%%%%%%%%%% ADAM %%%%%%%%%%")

    def train(self, weights, derivative_weights, biases, derivative_biases, iteration_number, **kwargs):
        """
        Adam algotihm to train a NeuralNetwork class.
        kwargs is added to admit natural arguments for SGD. Betweem those additional
        arguments, there must be one called epochs
        """
        # TAKING INTO ACCOUNT MOMENTUM HISTORY
        self.m_weights = [self.beta_1 * m_weight_layer + (1 - self.beta_1) * derivative_weight_layer
                          for m_weight_layer, derivative_weight_layer in zip(self.m_weights, derivative_weights)]
        self.v_weights = [self.beta_2 * v_weight_layer + (1 - self.beta_2) * np.power(derivative_weight_layer, 2)
                          for v_weight_layer, derivative_weight_layer in zip(self.v_weights, derivative_weights)]
        self.m_biases = [self.beta_1 * m_bias_layer + (1 - self.beta_1) * derivative_bias_layer
                         for m_bias_layer, derivative_bias_layer in zip(self.m_biases, derivative_biases)]
        self.v_biases = [self.beta_2 * v_bias_layer + (1 - self.beta_2) * np.power(derivative_bias_layer, 2)
                         for v_bias_layer, derivative_bias_layer in zip(self.v_biases, derivative_biases)]

        # EXPECTANCY CORRECTION
        m_weights_hat = [m_w_layer / (1 - np.power(self.beta_1, iteration_number)) for m_w_layer in self.m_weights]
        v_weights_hat = [v_w_layer / (1 - np.power(self.beta_2, iteration_number)) for v_w_layer in self.v_weights]
        m_biases_hat = [m_b_layer / (1 - np.power(self.beta_1, iteration_number)) for m_b_layer in self.m_biases]
        v_biases_hat = [v_b_layer / (1 - np.power(self.beta_2, iteration_number)) for v_b_layer in self.v_biases]

        # PARAMETER UPDATE
        weights = [weight_layer - self.learning_rate * m_layer / (np.sqrt(v_layer) + self.epsilon)
                   for weight_layer, m_layer, v_layer in zip(weights, m_weights_hat, v_weights_hat)]
        biases = [bias_layer - self.learning_rate * m_layer / (np.sqrt(v_layer) + self.epsilon)
                  for bias_layer, m_layer, v_layer in zip(biases, m_biases_hat, v_biases_hat)]

        return weights, biases
