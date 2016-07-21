__author__ = 'saftophobia'
import numpy as np
import logging

class ConvolutionalNeuralNetworks(object):
    def __init__(self, reg_lambda, learning_rate):
        self.W = 0
        self.b = 0
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate



    def L1(self, params):
        return np.sum(np.abs(params))

    def L2(self, params):
        return np.sum(params ** 2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_diff(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(x):
        return np.maximum(0.0, x)


    def relu_d(x):
        dx = np.zeros(x.shape)
        dx[x > 0] = 1
        return dx

    def reLU_soft(self, x):
        return np.log(1 + np.exp(x)) #log = ln

    def reLU_soft_diff(self,x):
        return self.sigmoid(x)

    def convolute(self, input, num_of_features):
        """ Return convoluted images
        :param input: output of previous layers with shape like 50,000 img X 3 channels X 1024 list
        :param num_of_features: number of features
        :return: convoluted images with shape (num_of_features x input_width x input high) e.g 16x32x32
        """
        batch_size = input.shape[0]
        img_channels = input.shape[1]
        img = input.shape[3]
        

    def subsample(self, convoluted_filters, pool_size, strides = 1, padding = 0, method = PoolingMethod.MEAN):
        """ Returns pooled images from convolutional filters
        :param convoluted_filters: output of previous convolutional layers e.g 16 layers x 32 width x 32 height
        :param pool_size: pool size e.g (2,2)
        :param strides: how many steps the filter should take in every iteration
        :param padding: start from inner frame of the image #NOT_IMPLEMENTED
        :param method: pooling methid is either PoolingMethods.MEAN or PoolingMethods.MAX
        :return: pooled images e.g 6 layers x 16x16
        """
        convoluted_num_filters = convoluted_filters.shape[0]
        convoluted_filter_width = convoluted_filters.shape[1]
        convoluted_filter_height = convoluted_filters.shape[2]

        pool_filter_width = pool_size.shape[0]
        pool_filter_height = pool_size.shape[1]

        pooled_img_width  = ((convoluted_filter_width + 2 * padding - pool_filter_width)/strides) + 1
        pooled_img_height = ((convoluted_filter_height + 2 * padding - pool_filter_height)/strides) + 1
        if (isinstance(pooled_img_width, int) and isinstance(pooled_img_height, int)):
            raise Exception("Invalid HyperParameters for subsample Layer ... ")

        pooled_imgs = np.zeros((convoluted_num_filters, pooled_img_width, pooled_img_height))

        for conv_filter in range(convoluted_num_filters):
            for row in range(pooled_img_width):
                for column in range(pooled_img_height):
                    r1 = row * pool_filter_width * strides
                    r2 = r1  + pool_filter_width
                    c1 = column * pool_filter_height * strides
                    c2 = column + pool_filter_height
                    pixels = convoluted_filters[conv_filter, r1:r2, c1:c2]
                    pooled_imgs[conv_filter, row, column] = PoolingMethod(pixels, method)

        return pooled_imgs


class PoolingMethod:
    MAX, MEAN = range(2)

    def pool_operation(self, pixels, method):
        if method == PoolingMethod.MAX : return np.max(pixels)
        if method == PoolingMethod.MEAN: return np.mean(pixels)
