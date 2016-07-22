__author__ = 'saftophobia'
import numpy as np
from util.helper import *

class Convolutional(object):
    def __init__(self, num_of_output_featureMaps, prev_layer_stack_size, filter_size, input_size, mini_batch_size, activation_function = reLU):
        self.filter_shape = 0
        self.num_of_output_featureMaps = num_of_output_featureMaps
        self.prev_layer_stack_size = prev_layer_stack_size # 6 layers x 16x16
        self.filter_size = filter_size
        self.input_size = input_size
        self.mini_batch_size = mini_batch_size
        self.activation_function = activation_function

        W_shape = (prev_layer_stack_size, num_of_output_featureMaps, filter_size, filter_size)
        self.W = np.random.normal(loc = 0, scale = 1, size = W_shape) # random bias for each filter! loc = mean of the distribution, scale = standard-deviation
        self.b = np.random.normal(loc = 0, scale = 1, size = num_of_output_featureMaps) # random bias for each filter! loc = mean of the distribution, scale = standard-deviation

    """
    :param input: 4 dim tuple => num_of_imgs (usually batch size), prev_layer_num_of_pooled_input, img_width, img_height
    """
    def forward(self, input):
        return convolute(input, num_of_output_featureMaps=self.num_of_output_featureMaps, filter_shape=(self.filter_size,self.filter_size), W = self.W, b = self.b, activation_function = self.activation_function)

    def backward(self): pass




class Pooling(object):
    def __init__(self, pool_size = 2, strides = 1, padding = 0, pooling_method = np.mean):
        self.size = 0
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.pooling_method = pooling_method

    def forward(self, input_feature_maps):
        pooled_out = subsample(input_feature_maps, self.pool_size)

        #import matplotlib.pyplot as plt
        #plt.imshow(pooled_out[1,2,])
        #plt.show()

        return pooled_out

class FullyConnected(object):
    def __init__(self):
        self.activation_fn = "ReLU"

