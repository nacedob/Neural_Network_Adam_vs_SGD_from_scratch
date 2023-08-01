import numpy as np
import matplotlib.pyplot as plt
import Functions
import InputData


# ------------------------------------------------------------------------------
# NEURAL NETWORK CLASS---------------------------------------------------------
# ------------------------------------------------------------------------------

class NeuralNetwork:
    """ Objects of class Neural Network represent neural networks, consisting
        of a input layer (whose size is input_size), an arbitrary number
        of hidden layers (number of neurons provided by hidden_sizes) and a
        ouput layer, whose size is given by output_size. The total number of layers
        is len(hidden_sizes)+2, denote by number_layers

        It has weights and biases separatley as parameters.
            - Weights: a list containing matrices (numpy arrays) for every layer
            - Biases: a list of numpy arrays for every layer

        This class contains several subroutines:

            - __init__ : constructor. You have to provided the arguments listed above
                    Also, it initializes randomly the weights and biases. Optionally, you can
                    the specify the activation function (defaults to sigmoid) to use and
                    its derivative; as well as cost function  (defaults to quadratic error).
            - feedforward: returns the prediction by the neural network of a given input
            - activation_function and derivative_function
            - cost function and its derivative
            - getActivations: returns the activations for every layers
            - backward_propagation: returns the partial derivative for the provided
                    cost function
            - train: uses above subroutines to directly train the network
                    given a training set. It requires an instance of an optimizer.
                    You can also decide if testing is
                    simultaneously done while training, and specify (or not) a
                    test set
            - test: in case test is asked, this test the current performance with
                    a provided test set. If no test set is provided, then uses
                    the training test to test it, but warns in the console
                    that it may no be accuarate  """

    def __init__(self, input_size, hidden_sizes, output_size,
                 activation_function_name="sigmoid", cost_function_name="quadratic_error"):

        """Constructor of neural net.

        Parameters:
        input_size (int): dimension of the input points
        hiddens_size (list): list containig at each index the number of neurons
                             for that layer. For example [2,7,3 ] will have three
                             hidden layers consisting of 2, 7 and 3 neurons
                             respectively
        output_size (int): dimension of the output points
        learning_rate (double): learning rate to train the net.
        activation_function_name (string): specify the activation function. It contains
                             the name of the method. Method must be defined
                             out of the class, with the exact name.
                             Defaults to sigmoid
        cost_function_name (string): specify the cost function. It contains
                             the name of the method. Method must be defined
                             out of the class, with the exact name.
                             Defaults to quadratic_error"""

        self.architecture = hidden_sizes.copy()
        self.architecture.insert(0, input_size)
        self.architecture.insert(len(self.architecture), output_size)

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.number_layers = len(self.architecture)

        # Initialize weights and biases for input layer to first hidden layer
        self.biases = [np.random.randn(layer, 1) for layer in self.architecture[1:]]
        self.weights = [np.random.randn(currentLayer, previousLayer)
                        for currentLayer, previousLayer
                        in zip(self.architecture[1:], self.architecture[:-1])]

        # STORE ACTIVATION AND COST FUNCTION NAME
        self.activation_function_name = activation_function_name
        self.cost_function_name = cost_function_name

        # IN CASE TEST DATA SET HAS BEEN PROVIDED, STORE PERFORMANCE EVOLUTION
        # WITH TRAINING EPOCHS
        self.performance = []

    def activation_function(self, x):
        """Activation function used for the NN. It is specified in the constructor.
        Used for the backwards and feedforward methods.
        Parameters
        ----------
        x : double / list
            Point where you want to compute the activation function.

        Returns
        -------
        TYPE: double/list
            Activation function at that point.
        """
        if self.activation_function_name == "sigmoid":
            # SIGMOID
            return Functions.sigmoid(x)

        elif self.activation_function_name == "tanh":
            return np.tanh(x)

    def activation_derivative(self, x):
        """
        Derivative of activation function used for the NN. It is specified in
        the constructor. Used for the backwards training

        Parameters
        ----------
        x : double / list
            Point where you want to compute the derivative of activation function.

        Returns
        -------
        TYPE: double/list
            Derivative of activation function at that point.
        """
        if self.activation_function_name == "sigmoid":
            # SIGMOID
            return Functions.sigmoid_derivative(x)

        elif self.activation_function_name == "tanh":
            return 1 - np.power(np.tanh(x), 2)

    def cost_function(self, prediction, real_output):
        """
        Activation function used for the NN. It is specified in the constructor.
        Used for the backwards and test methods.

        Parameters
        ----------
        prediction : double / list
            Prediction of our NN at some points.
        real_output : int / list
            Real known output from the training set

        Returns
        -------
        TYPE: double/list
            Activation function at that point.
        """

        if self.cost_function_name == "quadratic_error":
            # QUADRATIC ERROR
            return Functions.quadratic_error(prediction, real_output)
        # else:

    def cost_derivative(self, prediction, real_output):
        """
        Derivative of cost function used for the NN. It is specified in
        the constructor. Used for the backwards training

        Parameters
        ----------
        prediction : double / list
            Prediction of our NN at some points.
        real_output : int / list
            Real known output from the training set

        Returns
        -------
        TYPE: double/list
            Derivative of activation function at that point.
        """
        if self.cost_function_name == "quadratic_error":
            # QUADRATIC ERROR
            return Functions.quadratic_error_derivative(prediction, real_output)

    def feedforward(self, X):
        """
        Used to compute the ouput/prediction of the neural network

        Parameters
        ----------
        X : double/list
            Points where you want to predict  a result.

        Returns
        -------
        layerOutput : int/list
            Prediction / output of the neural network. Size of this list is
            specified in the constructor by output_size.
        """
        # Compute output for first hidden layer
        layer_output = self.activation_function(np.dot(self.weights[0], X) + self.biases[0])

        # Compute output for remaining hidden layers and output layer
        for i in range(1, len(self.weights)):
            layer_output = self.activation_function(np.dot(self.weights[i], layer_output) + self.biases[i])

        return layer_output

    def predict(self, x):
        """
        Uses feedforward and takes maximum index to make a prediction for image x
        Parameters
        ----------
        x : Numpy array (len=784)
            Input.

        Returns
        -------
        Int
            Prediction for x.

        """
        return np.argmax(self.feedforward(x))

    def probabilities(self, x):
        """
        Uses feedforward to compute probabilities

        Parameters
        ----------
        x : Numpy array (len=784)
            Input.

        Returns
        -------
        Numpy array (len=10)
            Probabilities for each character.

        """
        aux = self.feedforward(x)
        return aux / max(aux)

    def backward_propagation(self, x, y):
        """
        This method takes a SINGLE training point and computes its contribution
        to partial derivatives using backwards propagation

        Parameters
        ----------
        x : double
            Input part of a SINGLE training point.
        y : double
            Output  part of a SINGLE training point (output computed by net).

        Returns
        -------
        derivative_bias_weight : List of numpy arrays
            Partial derivatives wrt the bias of cost function evaluated for the
            particular point.
        derivative_bias_point : List of numpy arrays
            Partial derivatives wrt the weitght of cost function evaluated for
            the particular point.
        """
        # INITIALIZATION
        derivative_bias_point = [np.zeros(bias.shape) for bias in self.biases]
        derivative_bias_weight = [np.zeros(weight.shape) for weight in self.weights]
        # feedforward
        activations = [x]  # list to store all the activations, layer by layer
        transfers = []  # list to store all the transfer vectors, layer by layer

        # COMPUTE LIST OF ACTIVATIONS AND TRANSFERS
        for bias, weight in zip(self.biases, self.weights):
            transfer = np.dot(weight, activations[-1]) + bias
            transfers.append(transfer)
            activations.append(self.activation_function(transfer))

        # backward pass for last layer
        delta = self.cost_derivative(activations[-1], y) * self.activation_derivative(
            transfers[-1])  # delta de la ultima capa
        derivative_bias_point[-1] = delta
        derivative_bias_weight[-1] = np.dot(delta, activations[-2].transpose())

        # backwards for the rest of the layers
        for layer in range(2, self.number_layers):
            # Note that the variable layer in the loop below is used a little
            # differently to the notation in Chapter 2 of the book.  Here,
            # layer = 1 means the last layer of neurons, layer = 2 is the
            # second-last layer, and so on.  It's a renumbering of the
            # scheme in the book, used here to take advantage of the fact
            # that Python can use negative indices in lists.
            transfer = transfers[-layer]
            delta = np.dot(self.weights[-layer + 1].T, delta) * self.activation_derivative(transfer)
            derivative_bias_point[-layer] = delta
            derivative_bias_weight[-layer] = np.dot(delta, activations[-layer - 1].T)

        return derivative_bias_weight, derivative_bias_point

    def train(self, optimizer, training_data, number_epochs, mini_batch_size,
              test_data=None, plot_test=False):
        """
        Used for training the neural network

        Parameters
        ----------
        optimizer: Optimizer
            Optimizer used to train net
        training_data: list
            Training data consisting in inputs and expected ouputs
        number_epochs : int
            Represents the number of iterations made to train the neural network.
        mini_batch_size: int
            mini batch sized used to train net
        test_data : boolean, optional
            Expresses if you want to test the neural network during training.
            The default is False.
        plot_test : boolean, optional
            Expresses if you want to see graphically performance ot the neural
            network during training. It cannot be True in case test_date is
            False. The default is False.

        Returns
        -------
        None.
        """
        if test_data:
            test_data_length = len(test_data)
            epoch_performance = self.check_performance(test_data)
            self.performance.append(epoch_performance)
            print("Performance for epoch 0: {0} %".format(round(100 * epoch_performance / test_data_length, 2)))

        for epoch in range(number_epochs):

            splitted_in_mini_batches = InputData.mini_batch_data(training_data, mini_batch_size)

            for mini_batch in splitted_in_mini_batches:

                # BACKWARD PROPAGATION
                derivative_weights = [np.zeros(w.shape) for w in self.weights]
                derivative_biases = [np.zeros(b.shape) for b in self.biases]
                for x, y in mini_batch:
                    derivative_weights_point, derivative_biases_point = self.backward_propagation(x, y)

                    derivative_weights = [global_der + point_der for global_der, point_der
                                          in zip(derivative_weights, derivative_weights_point)]
                    derivative_biases = [global_der + point_der for global_der, point_der
                                         in zip(derivative_biases, derivative_biases_point)]

                # (STOCHASTIC) GRADIENT DESCENT -> sum all points taking advantange of cost functions

                self.weights, self.biases = \
                    optimizer.train(self.weights, derivative_weights, self.biases, derivative_biases,
                                    iteration_number=epoch + 1, mini_batch_size=mini_batch_size)

            if test_data:
                epoch_performance = self.check_performance(test_data)
                self.performance.append(epoch_performance)
                print("Performance for epoch {0}: {1} %".format(epoch + 1,
                                                                round(100 * epoch_performance / test_data_length, 2)))
            else:
                print(f"Epoch {epoch} complete")
        if plot_test:
            self.plot_performance(test_data_length)

    def check_performance(self, test_data):
        """
        Test the parameters to see its eficiency

        Parameters
        ----------
        test_data : list of numpy arrays
            Training set.

        Returns
        -------
        String
            String that contains the percentage of correctly solved points.
            When using it, you have to replace the "i_replace" by the number
            iteration, so you can display it on the console
        """
        test_results = [(self.predict(x), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def plot_performance(self, top):
        """
        Function to plot the evolution of performance of neural network

        Parameters
        ----------
        top : int
            Number of training points.

        Returns
        -------
        None.
        """

        x_axis = list(range(0, len(self.performance)))
        top_perfomance_line = top * np.ones(len(self.performance), list)

        performance_plot = plt.figure()
        plt.figure(performance_plot)
        plt.plot(x_axis, self.performance, 'o-.')
        plt.plot(x_axis, top_perfomance_line, ":k")
        plt.title("# cifras bien clasificadas")
        plt.xlabel("Épocas")
        plt.ylabel("Aciertos")
        plt.show()
