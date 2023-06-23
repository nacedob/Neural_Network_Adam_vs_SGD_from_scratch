import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#------------------------------------------------------------------------------
# DEFINITION OF POSSIBLE COST FUNCTIONS----------------------------------------
#------------------------------------------------------------------------------

def quadratic_error(prediction, real_output) :
    return np.sum(np.power(prediction-real_output, 2))/len(prediction)

def quadratic_error_derivative(prediction, real_output):   
    return prediction - real_output

#------------------------------------------------------------------------------
# DEFINITION OF POSSIBLE ACTIVATION FUNCTIONS----------------------------------
#------------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return np.exp(-x)*np.square(sigmoid(x))

#------------------------------------------------------------------------------
# NEURAL NETWORK CLASS---------------------------------------------------------
#------------------------------------------------------------------------------

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
    
    def __init__(self, input_size, hidden_sizes, output_size,\
                 activation_function_name = "sigmoid", cost_function_name = "quadratic_error"):
        
        """Constructor of neural net.

        Parameters:
        input_size (integer): dimension of the input points 
        hiddens_size (list): list containig at each index the number of neurons
                             for that layer. For example [2,7,3 ] will have three 
                             hidden layers consisting of 2, 7 and 3 neurons 
                             respectively
        output_size (integer): dimension of the output points
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
        self.biases = [np.random.randn(layer,1) for layer in self.architecture[1:]]
        self.weights = [np.random.randn(currentLayer,previousLayer) 
                            for currentLayer, previousLayer 
                                in zip(self.architecture[1:],self.architecture[:-1])]
        
        #STORE ACTIVATION AND COST FUNCTION NAME
        self.activation_function_name = activation_function_name
        self.cost_function_name = cost_function_name

        #IN CASE TEST DATA SET HAS BEEN PROVIDED, STORE PERFORMANCE EVOLUTION 
        #WITH TRAINING EPOCHS
        self.performance = []


    def activation_function(self,x):
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
            return sigmoid(x)
        
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
            return sigmoid_derivative(x)
        
        elif self.activation_function_name == "tanh":
            return 1-np.power(np.tanh(x),2)
        
        
    def cost_function(self,prediction, real_output):
        """
        Activation function used for the NN. It is specified in the constructor.
        Used for the backwards and test methods.
        
        Parameters
        ----------
        prediction : double / list
            Prediction of our NN at some points.
        real_output : integer / list
            Real known output from the training set
            
        Returns
        -------
        TYPE: double/list
            Activation function at that point.
        """
        
        if self.cost_function_name == "quadratic_error":
            # QUADRATIC ERROR
            return quadratic_error(prediction, real_output)
        #else:
            

    def cost_derivative(self, prediction, real_output):
        """
        Derivative of cost function used for the NN. It is specified in 
        the constructor. Used for the backwards training
        
        Parameters
        ----------
        prediction : double / list
            Prediction of our NN at some points.
        real_output : integer / list
            Real known output from the training set

        Returns
        -------
        TYPE: double/list
            Derivative of activation function at that point.
        """
        if self.cost_function_name == "quadratic_error":
            # QUADRATIC ERROR
            return quadratic_error_derivative(prediction, real_output)


    def feedforward(self, X):
        """
        Used to compute the ouput/prediction of the neural network

        Parameters
        ----------
        X : double/list
            Points where you want to predict  a result.

        Returns
        -------
        layerOutput : integer/list
            Prediction / output of the neural network. Size of this list is 
            specified in the constructor by output_size.
        """
        # Compute output for first hidden layer
        layerOutput = self.activation_function(np.dot(self.weights[0],X) + self.biases[0])

        # Compute output for remaining hidden layers and output layer
        for i in range(1, len(self.weights)):
            layerOutput = self.activation_function(np.dot(self.weights[i],layerOutput) + self.biases[i])

        return layerOutput
    
    
    def predict(self,x):
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
    
    
    def probabilities(self,x):
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
        return aux/max(aux)
    
    
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
        #INITIALIZATION
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
        delta = self.cost_derivative(activations[-1], y) * self.activation_derivative(transfers[-1])  # delta de la ultima capa
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
            delta = np.dot(self.weights[-layer+1].T, delta) * self.activation_derivative(transfer)
            derivative_bias_point[-layer] = delta
            derivative_bias_weight[-layer] = np.dot(delta, activations[-layer-1].T)
        
        return (derivative_bias_weight, derivative_bias_point)


    def train(self, optimizer, training_data, number_epochs, mini_batch_size,
              test_data = None, plot_test = False):
        """
        Used for training the neural network

        Parameters
        ----------
        training_data: list
            Training data consisting in inputs and expected ouputs
        number_epochs : integer
            Represents the number of iterations made to train the neural network.
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
            print("Performance for epoch 0: {0} %".format(round(100*epoch_performance/test_data_length,2)))
        
        for epoch in range(number_epochs):
            
            splitted_in_mini_batches = Input.mini_batch_data(training_data, mini_batch_size)
            
            for mini_batch in splitted_in_mini_batches:
                 
                # BACKWARD PROPAGATION
                derivative_weights = [np.zeros(w.shape) for w in self.weights]
                derivative_biases = [np.zeros(b.shape) for b in self.biases]
                for x,y in mini_batch:
                    derivative_weights_point, derivative_biases_point = self.backward_propagation(x,y)
                    
                    derivative_weights = [global_der + point_der for global_der,point_der
                                          in zip(derivative_weights,derivative_weights_point)]
                    derivative_biases = [global_der + point_der for global_der,point_der
                                          in zip(derivative_biases,derivative_biases_point)]
                    
                # (STOCHASTIC) GRADIENT DESCENT -> sum all points taking advantange of cost functions 

                self.weights, self.biases = \
                    optimizer.train(self.weights, derivative_weights, self.biases, derivative_biases, iteration_number=epoch+1, mini_batch_sizeini_batch_size=mini_batch_size) 
                
            if test_data:
                epoch_performance = self.check_performance(test_data)
                self.performance.append(epoch_performance)
                print("Performance for epoch {0}: {1} %".format(epoch+1,round(100*epoch_performance/test_data_length,2)))
            else:
                print(f"Epoch {epoch} complete")
        if plot_test == True:
            self.plot_performance(test_data_length)
            

    def check_performance(self, test_data):
        """
        Test the parameters to see its eficiency

        Parameters
        ----------
        test_x : list of numpy arrays
            input points of a training set.
        test_y : list of numpy arrays
            Output points of a training set.
        plot_statistics : boolean, optional
            Expresses if you want to see a plot to see graphically the 
            performance. The default is False.

        Returns
        -------
        String
            String that contains the percentage of correctly solved points. 
            When using it, you have to replace the "i_replace" by the number
            iteration, so you can display it on the console
        """
        test_results = [(self.predict(x), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
        
    def plot_performance(self,top):
        """
        Function to plot the evolution of performance of neural network

        Parameters
        ----------
        top : integer
            Number of training points.

        Returns
        -------
        None.
        """
        
        xAxis = list(range(1,len(self.performance)+1))
        topPerfomanceLine = top* np.ones(len(self.performance),list)
        
        performancePlot = plt.figure()
        plt.figure(performancePlot)
        plt.plot(xAxis, self.performance,'o-.')
        plt.plot(xAxis, topPerfomanceLine,":k")
        plt.title("# cifras bien clasificadas")
        plt.xlabel("Épocas")
        plt.ylabel("Aciertos")
        plt.show()

#------------------------------------------------------------------------------
# OPTIMIZERS-------------------------------------------------------------------
#------------------------------------------------------------------------------

from abc import ABC, abstractmethod

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
        self.learning_rate=learning_rate
        # self.mini_batch_size=mini_batch_size
        
    def train(self, weights, derivative_weights, biases, derivative_biases, **kwargs):
        """
        Stochastic gradient descent to train a NeuralNetwork class. 
        kwargs is added to admit natural arguments for Adam. Betweem those additional
        arguments, there must be one called mini_batch_size
        """

        # SGD update
        weights = [weight_layer-self.learning_rate*derivative_weight_layer/mini_batch_size 
                   for weight_layer,derivative_weight_layer in zip(weights,derivative_weights)]
        biases = [bias_layer--self.learning_rate*derivative_bias_layer/mini_batch_size 
                  for bias_layer,derivative_bias_layer in zip(biases,derivative_biases)]
             
        return weights,biases
    
class Adam(Optimizer):
    
    def __init__(self, architecture, learning_rate = 1e3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
        """
        Adam constructor. Specify learning_rate, beta_1, beta_2 and epsilon
        """
        
        # Initizalize momentums
        self.m_biases = [np.zeros([layer,1]) for layer in architecture[1:]]
        self.v_biases = [np.zeros([layer,1]) for layer in architecture[1:]]
        self.m_weights = [np.zeros([currentLayer,previousLayer]) 
                            for currentLayer, previousLayer 
                                in zip(architecture[1:],architecture[:-1])]
        self.v_weights = [np.zeros([currentLayer,previousLayer]) 
                            for currentLayer, previousLayer 
                                in zip(architecture[1:],architecture[:-1])]
        
        # Hiperparameters
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
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
        m_biases_hat = [m_b_layer/ (1 - np.power(self.beta_1, iteration_number)) for m_b_layer in self.m_biases]
        v_biases_hat = [v_b_layer / (1 - np.power(self.beta_2, iteration_number)) for v_b_layer in self.v_biases]
        
        #PARAMETER UPDATE
        weights = [ weight_layer- self.learning_rate * m_layer / (np.sqrt(v_layer) + self.epsilon)
                      for weight_layer, m_layer, v_layer in zip(weights, m_weights_hat, v_weights_hat)]
        biases = [ bias_layer- self.learning_rate * m_layer / (np.sqrt(v_layer) + self.epsilon)
                      for bias_layer, m_layer, v_layer in zip(biases, m_biases_hat, v_biases_hat)]
        
        return weights, biases
    

#------------------------------------------------------------------------------
# METHODS TO LOAD DATA BASE AND TRANSFORM PICTURES-----------------------------
#------------------------------------------------------------------------------

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
        training_data, validation_data, test_data = pickle.load(f,encoding = 'iso-8859-1')
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
        array = np.array(picture, dtype = np.float32)
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
        np.random.shuffle(training_data) #rearrange the training_data randomly
        mini_batches = [ training_data[k:k + mini_batch_size] 
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
            random_data = training_data [0:mini_batch_size]
        return random_data


#------------------------------------------------------------------------------
# MAIN PROGRAM ----------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == "__main__":
    
    import time
    # Load training and test data
    training_data, validation_data, test_data = Input.load_mnist_database()
    
    # Create instances of NeuralNetwork class to compare SGD and Adam
    clasificadorDigitosSGD = NeuralNetwork(784, [28], 10)
    clasificadorDigitosAdam = NeuralNetwork(784, [28], 10)
    clasificadorDigitosAdam.weights=clasificadorDigitosSGD.weights
    clasificadorDigitosAdam.biases=clasificadorDigitosSGD.biases
    
    # Comom hiperparameters
    number_epochs = 8
    mini_batch_size = 20
    
    # Training with Stochastic Gradient Descent
    sgd = SGD(learning_rate = 8)
    start_time = time.time()
    clasificadorDigitosSGD.train(sgd, training_data, number_epochs, mini_batch_size, 
                                 test_data = test_data, plot_test = False)
    end_time = time.time()
    timeSgd=round(end_time-start_time,2)
    print(f"-----------Training finished-----------\n Spent time: \
          {round(end_time-start_time,2)} segs")
    
    # Training with Stochastic Gradient Descent
    adam = Adam(clasificadorDigitosAdam.architecture,
                                 learning_rate=0.017, beta_1=0.86, beta_2=0.991, epsilon=1e-6)
    start_time = time.time()
    clasificadorDigitosAdam.train(adam, training_data, number_epochs, 
                                  mini_batch_size, test_data = test_data, plot_test = False)
    end_time = time.time()
    timeAdam=round(end_time-start_time,2)
    print(f"-----------Training finished-----------\n Spent time: \
          {round(end_time-start_time,2)} segs")
    
                               
    # -------PLOTS---------
    performancePlot = plt.figure()
    performancesSGD=100*np.array(clasificadorDigitosAdam.performance)/len(test_data)
    performancesAdam=100*np.array(clasificadorDigitosAdam.performance)/len(test_data)
    
    plt.figure()
    width = 0.1
    offset=0.15
    xAxis = list(range(len(performancesAdam)))
    xAxis2 = [x+offset for x in xAxis]
    plt.bar(xAxis,  performancesSGD, width=width, align='center', color='blue', label="SGD")
    plt.bar(xAxis2, performancesAdam, width=width, align='center', color= 'red', label="Adam")
    plt.xticks(np.arange(0,number_epochs+1,1))
    plt.yticks(np.arange(0,110,10))
    plt.xlabel("Épocas")
    plt.ylabel("Precisión (%)")
    plt.legend(loc='upper left',bbox_to_anchor=(1, 1))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.grid(axis="y",linestyle="--")
    plt.grid(axis="y",which="minor", linestyle=":")
    plt.show()
    