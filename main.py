__author__ = 'saftophobia'

from network import *
from data.CIFAR10 import CIFAR10
from util import util, logger

dataset = CIFAR10(5)

#print dataset.TAGS[dataset.labels[1]]
#util.draw_image(dataset.data[1])


v = ConvolutionalNeuralNetworks(layers=[
        Convolutional(num_of_output_featureMaps = 16,
                       prev_layer_stack_size = 3,
                       filter_size = 5,
                       input_size = 32,
                       mini_batch_size= 64)
        #Pooling()
                        ])
v.train(dataset.data[1:15], dataset.labels[1:15])

#num_of_imgs (usually batch size), prev_layer_num_of_pooled_input, img_width, img_height

mini_batch_size=4
num_prev_layer_channels = 3
img_width = 32
img_height = 32


#conv2d.forward(input=dataset.data[0:4])