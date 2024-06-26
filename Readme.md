# Neural-Network-TFG
 
This code has been created as a Final Bachelor Project (Double Degree in Mathematics and Physics - Universidad Complutense de Madrid), or TFG in my original language -Spanish.

I have created this code on my own, based on Michael Nielsen's idea. The main objective is to be able to create a Neural Network able to classify handwritten digits. For that purpose, I have chosen MNIST database to carry out network training and to test performance.

The main idea is to compare theoretical  and experimentally the use of different Optimizers to train networks. In this case, I have chosen to train our neural network first with classic Stochastic Gradient Descent (SGD) and [Adam](https://arxiv.org/abs/1412.6980).

As demonstrate in my project, in both cases, Adam over-performs SGD [theoretical  proof in pdf](Bachelor Final Thesis.pdf). You can check experimental proof here in python code, just by running main file. This experiment is explained later.

## Project structure

Project is structured as follows:
- [Functions.py](Functions.py): this file encapsulates miscellaneous function to be used, mainly, activation functions and cost function.
- [InpoutData.py](InputData.py): this file encapsulates methods to load MNIST database or transform a input file to be able to be ingested by net.
- [Main.py](Main.py): carries out experiment to compare Adam and SGD performances
- [mnist.pkl.gz](mnist.pkl.gz): mnist database compressed
- [NeuralNetwork.py](NeuralNetwork.py): contains a class representing Neural Networks, that implements lots of functions to train, check performance, feedforward...
- [Optimizers.py](Optimizers.py):  file that contains an interface (ABC class) that represents a optimizer. This class is inherited by SGD and Adam Optimizer 
- [Bachelor_Final_Thesis.pdf](Bachelor_Final_Thesis.pdf): Final Bachelor Project presented, where you can learn about basics of Neural Networks studied from a mathematical point of view. Here, some details of this code are explained as well. Unfortunately, as I am Spanish and so do my university, it is written in my mother tongue. Nevertheless, it is always a good idea to start learning other languages ;)

## Experiment: SGD vs Adam

To compare SGD and Adam, I will instantiate a NeuralNetwork object and make a copy of it. One of these objects will be trained using SGD, while the other will be trained using Adam. Additionally, you can monitor the network's performance for each epoch in the console, and after the program concludes, you can view a bar graph comparing the results.

As you can see in (my bachelor thesis)[Bachelor Final Thesis.pdf], Adam outperforms  SGD. Specifically, I have achieved a 95% precision with Adam and a 94% with SGD, so the difference is not so big. On the other hand, regarding total time spent by both nets to be trained, you can check that SGD is faster. Thus, it can be inferred that SGD might be more suitable for cases involving very heavy training, where precision is not significantly compromised.
