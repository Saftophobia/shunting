__author__ = 'saftophobia'
import numpy as np
import logging
from layers import *

class ConvolutionalNeuralNetworks(object):
    def __init__(self, layers, regularizer_lambda = 0.1, learning_rate = 0.01, mini_batch_size = 32, max_iterations = 10):
        self.layers = layers
        self.reg_lambda = regularizer_lambda
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.max_iterations = max_iterations

    # data = 50,000 x 3 x 32 x 32
    # labels = 50,000
    def train(self, data, labels):
        num_batches = data.shape[0] // self.mini_batch_size
        for iteration in range(self.max_iterations):
            for batch_number in range(self.mini_batch_size):
                data_batch   = data[batch_number * self.mini_batch_size : (batch_number + 1) * self.mini_batch_size , ]
                labels_batch = labels[batch_number * self.mini_batch_size : (batch_number + 1) * self.mini_batch_size]

                #forward
                for layer in self.layers:
                    data_batch = layer.forward(data_batch)
                f_predicted = data_batch

                #calculate misclassifications in the softmaxLayer
                sgd = self.layers[-1].backward(self.format_labels(labels_batch, f_predicted), f_predicted)

                #backward - pass error to other layers
                for layer in reversed(self.layers[:-1]):
                    sgd = layer.backward(sgd)


                #update params

                logging.info("Iteration %i, \t batch number: %i completed!" % (iteration, batch_number))

                error = self.calculate_error(labels_batch, f_predicted)
            #logging.info("Iteration %i, loss %.3f" % (iteration, loss))

    def format_labels(self, labels, predicted):
        modified_labels = np.zeros(predicted.shape)
        for record in range(labels.shape[0]):
            modified_labels[record, labels[record] - 1] = 1 #change label 3 to (0,0,1,0 .. )


        return modified_labels

