__author__ = 'saftophobia'

from network import *
from data.CIFAR10 import CIFAR10
from util import util, logger

LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

dataset = CIFAR10(5)
util.shuffle_data(dataset)


print LABELS[dataset.labels[1]]
util.draw_image(dataset.data[1])

convnn = ConvolutionalNeuralNetworks(reg_lambda = 0.1, learning_rate = 0.01)

convnn.convolute(dataset.data, 32)