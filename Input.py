# ------------------------------------------------------------------------------
# METHODS TO LOAD DATA BASE AND TRANSFORM PICTURES-----------------------------
# ------------------------------------------------------------------------------

import numpy as np


class Input:
    """
    Class Input contains static methods to load training, validation and test sets
    as well as convert any picture to a Numpy array of len = 784
    """

    @staticmethod
    def load_mnist_database():
        """
        Loads the MNIST database splitted in 3 subsets: training, test and
        validation. This data is also treated to be used for our NeuralNetwork.
        Adapted from M.Nielsen GitHub project

        Returns
        -------
        training_data, validation_data, test_data
        """

        import sys, pickle, gzip
        sys.path.append("./")
        f = gzip.open('mnist.pkl.gz', 'rb')
        training_data, validation_data, test_data = pickle.load(f, encoding='iso-8859-1')
        f.close()
        training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
        training_results = [Input.vectorized_result(y) for y in training_data[1]]
        training_data = list(zip(training_inputs, training_results))
        validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
        validation_data = list(zip(validation_inputs, validation_data[1]))
        test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
        test_data = list(zip(test_inputs, test_data[1]))
        return (training_data, validation_data, test_data)

    @staticmethod
    def vectorized_result(j):
        """
        Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.
        """

        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    @staticmethod
    def picture2array(path):
        """
        Takes a picture in any possible format and converts it to be loaded
        by our Neural Network.
        """
        from PIL import Image

        picture = Image.open(path).convert('L').resize((28, 28))
        array = np.array(picture, dtype=np.float32)
        array /= 255.0
        return array.reshape(784, 1)

    @staticmethod
    def mini_batch_data(training_data, mini_batch_size):
        """
        Used create mini-batch sets used to train neural network in Stochastic
        Gradient Descent.
        This function splits the ENTIRE training sets randomly sorted and packed
        in mini_batches

        Parameters
        ----------
        training_data: list
            Training data consisting in inputs and expected ouputs
        mini_batch_size :
            Size of the created mini batch

        Returns
        -------
        mini_batches: list
            List of mini-batches
        """
        np.random.shuffle(training_data)  # rearrange the training_data randomly
        mini_batches = [training_data[k:k + mini_batch_size]
                        for k in range(0, len(training_data), mini_batch_size)]
        return mini_batches

    @staticmethod
    def alternative_mini_batch_data(training_data, mini_batch_size):
        """
        Used create mini-batch sets used to train neural network in Stochastic
        Gradient Descent.
        This function just creates a set of size mini-batch randomly out of the
        entire training set

        Parameters
        ----------
        training_data: list
            Training data consisting in inputs and expected ouputs
        mini_batch_size :
            Size of the created mini batch

        Returns
        -------
        random_data: list
            Random set
        """
        import random
        # CHOOSE MINI_BATCH_SIZE RANDOMLY
        if mini_batch_size > len(training_data):
            random_data = training_data
        else:
            random.shuffle(training_data)
            random_data = training_data[0:mini_batch_size]
        return random_data
