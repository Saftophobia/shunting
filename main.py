__author__ = 'saftophobia'

from network import *
from data.CIFAR10 import CIFAR10
from util import util, logger

dataset = CIFAR10(5)

#print dataset.TAGS[dataset.labels[1]]
#util.draw_image(dataset.data[1])

v = ConvolutionalNeuralNetworks(layers=[
        ConvolutionalLayer(num_of_output_featureMaps = 16,
                       prev_layer_stack_size = 3,
                       filter_size = 5,
                       mini_batch_size= 64),
        ActivationLayer(),
        PoolingLayer(),

        ConvolutionalLayer(num_of_output_featureMaps = 20,
                       prev_layer_stack_size = 16,
                       filter_size = 5,
                       mini_batch_size= 64),
        ActivationLayer(),
        PoolingLayer(),

        FullyConnectedLayer(prev_stack_size = 20,
                       output_size = 10)
                        ])


v.train(dataset.data, dataset.labels)

#num_of_imgs (usually batch size), prev_layer_num_of_pooled_input, img_width, img_height
#conv2d.forward(input=dataset.data[0:4])