import numpy as np
import InputData
from NeuralNetwork import NeuralNetwork
import Optimizers
from Functions import Sigmoid, QuadracticError
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if __name__ == "__main__":
    import time

    # Load training and test data
    training_data, validation_data, test_data = InputData.load_mnist_database()

    # Create instances of NeuralNetwork class to compare SGD and Adam
    digitClassificatorSGD = NeuralNetwork(784, [28], 10)
    digitClassificatorAdam = NeuralNetwork(784, [28], 10)
    digitClassificatorAdam.weights = digitClassificatorSGD.weights
    digitClassificatorAdam.biases = digitClassificatorSGD.biases

    # Define Cost and Activation Function
    sigmoid = Sigmoid()
    quadratic_cost = QuadracticError()

    # Common hiperparameters
    number_epochs = 8
    mini_batch_size = 20

    # Training with Stochastic Gradient Descent
    sgd = Optimizers.SGD(learning_rate=8)
    start_time = time.time()
    digitClassificatorSGD.train(sgd, training_data, number_epochs, mini_batch_size,
                                test_data=test_data, plot_test=False)
    end_time = time.time()
    timeSgd = round(end_time - start_time, 2)
    print(f"-----------Training finished-----------\n Spent time: \
          {round(end_time - start_time, 2)} segs")

    # Training with Stochastic Gradient Descent
    adam = Optimizers.Adam(digitClassificatorAdam.architecture,
                           learning_rate=0.017, beta_1=0.86, beta_2=0.991, epsilon=1e-6)
    start_time = time.time()
    digitClassificatorAdam.train(adam, training_data, number_epochs,
                                 mini_batch_size, test_data=test_data, plot_test=False)
    end_time = time.time()
    timeAdam = round(end_time - start_time, 2)
    print(f"-----------Training finished-----------\n Spent time: \
          {round(end_time - start_time, 2)} segs")

    # -------PLOTS---------
    performancePlot = plt.figure()
    performancesSGD = 100 * np.array(digitClassificatorSGD.performance) / len(test_data)
    performancesAdam = 100 * np.array(digitClassificatorAdam.performance) / len(test_data)

    plt.figure()
    width = 0.1
    offset = 0.15
    xAxis = list(range(len(performancesAdam)))
    xAxis2 = [x + offset for x in xAxis]
    plt.bar(xAxis, performancesSGD, width=width, align='center', color='blue', label="SGD")
    plt.bar(xAxis2, performancesAdam, width=width, align='center', color='red', label="Adam")
    plt.xticks(np.arange(0, number_epochs + 1, 1))
    plt.yticks(np.arange(0, 110, 10))
    plt.xlabel("Épocas")
    plt.ylabel("Precisión (%)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.grid(axis="y", linestyle="--")
    plt.grid(axis="y", which="minor", linestyle=":")
    plt.show()
