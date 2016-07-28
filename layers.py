__author__ = 'saftophobia'
import numpy as np
from util.helper import *

class ConvolutionalLayer(object):
    def __init__(self, num_of_output_featureMaps, prev_layer_stack_size, filter_size, mini_batch_size):
        self.filter_shape = 0
        self.num_of_output_featureMaps = num_of_output_featureMaps
        self.prev_layer_stack_size = prev_layer_stack_size # 6 layers x 16x16
        self.filter_size = filter_size
        self.mini_batch_size = mini_batch_size

        W_shape = (prev_layer_stack_size, num_of_output_featureMaps, filter_size, filter_size)
        self.W = np.random.normal(loc = 0, scale = 0.1, size = W_shape) # random bias for each filter! loc = mean of the distribution, scale = standard-deviation
        self.b = np.random.normal(loc = 0, scale = 0.1, size = num_of_output_featureMaps) # random bias for each filter! loc = mean of the distribution, scale = standard-deviation

    def forward(self, input_img):
        """ Return convoluted images
        :param input_img: input has shape (mini_batch_size, prev_layer_stack_size, img_width, img_height)
        :param num_of_output_featureMaps: number of features
        :param W: weight ndarray (shared) size (prev_layer_stack_size, num_of_output_featureMaps, filter_size, filter_size)
        :param b: bias array (shared) size = num_of_features
        :return feature_maps: convoluted images with shape (num_images x num_of_features x input_width x input high) e.g 100x 16x32x32
        """
        num_input_pooled_imgs = input_img.shape[0] # or channels if it's the first layer
        prev_layer_stack_size = input_img.shape[1]
        input_img_width = input_img.shape[2]
        input_img_height = input_img.shape[3]

        feature_maps = np.zeros((num_input_pooled_imgs, self.num_of_output_featureMaps, input_img_width, input_img_height))

        for img in range(num_input_pooled_imgs):
            for featureMap_num in range(self.num_of_output_featureMaps):
                feature_map = np.zeros((input_img_width, input_img_height))

                for input_num in range(prev_layer_stack_size):
                    image = input_img[img, input_num, : , :]
                    filter = self.W[input_num, featureMap_num, ] #filter for each pooled image in the stack

                    feature_map = feature_map + scipy.ndimage.filters.convolve(image, filter, mode='reflect') #conv

                feature_map_w_bias = feature_map + self.b[featureMap_num]
                feature_maps[img, featureMap_num, :, : ] = feature_map_w_bias

        return feature_maps


    def backward(self): pass


class ActivationLayer(object):
    def __init__(self, function = reLU):
        self.function = function

    def forward(self, input):
        self.input = input
        return self.function(input)

    def backward(self, prev_sgd):
        if (self.function == reLU): return prev_sgd * reLU_diff(self.input)
        elif (self.function == sigmoid): return prev_sgd * sigmoid_diff(self.input)
        elif (self.function == reLU_soft): return prev_sgd * reLU_soft_diff(self.input)


class PoolingLayer(object):
    def __init__(self, pool_size = 2, strides = 1, padding = 0, pooling_method = np.mean):
        self.size = 0
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.pooling_method = pooling_method

    def forward(self, input_feature_maps):
        #print input_feature_maps.shape #(32, 16, 32, 32)
        """ Returns pooled images from convolutional filters
        :param input_feature_maps: output of previous convolutional layers e.g 32 img x 16 featureMap x 32 width x 32 height
        :param pool_size: pool size e.g (2,2)
        :param strides: how many steps the filter should take in every iteration
        :param padding: start from inner frame of the image #NOT_IMPLEMENTED
        :param method: pooling methid is either PoolingMethods.MEAN or PoolingMethods.MAX
        :return pooled_imgs: pooled images e.g 32img x 6 layers x 16x16
        """
        self.input = input_feature_maps

        num_images = input_feature_maps.shape[0]
        convoluted_num_featureMaps = input_feature_maps.shape[1]
        convoluted_filter_width = input_feature_maps.shape[2]
        convoluted_filter_height = input_feature_maps.shape[3]

        pooled_img_width  = ((convoluted_filter_width + 2 * self.padding - self.pool_size)/self.strides) + 1
        pooled_img_height = ((convoluted_filter_height + 2 * self.padding - self.pool_size)/self.strides) + 1
        if not (float(pooled_img_width).is_integer() and float(pooled_img_height).is_integer()):
            raise Exception("Invalid HyperParameters for subsample Layer! pooled image width/height is %.4f" % pooled_img_width)

        pooled_imgs = np.zeros((num_images, convoluted_num_featureMaps, pooled_img_width, pooled_img_height))

        maximum_mean_map = np.zeros(self.input.shape)

        for img in range(num_images):
            for conv_filter in range(convoluted_num_featureMaps):
                for row in range(pooled_img_width):
                    r1 = row * self.strides
                    r2 = r1  + self.pool_size
                    for column in range(pooled_img_height):
                        c1 = column * self.strides
                        c2 = c1 + self.pool_size
                        pixels = input_feature_maps[img, conv_filter, r1:r2, c1:c2]

                        if self.pooling_method == np.mean:
                            mean = np.sum(pixels, dtype='double')
                            if mean != 0:
                                maximum_mean_map[img, conv_filter, r1:r2, c1:c2] = np.array(pixels)/np.sum(pixels, dtype='double')
                            else:
                                maximum_mean_map[img, conv_filter, r1:r2, c1:c2] = np.array(pixels)
                        else:
                            pixels_zeros = np.zeros_like(pixels)
                            pixels_zeros[np.arange(len(pixels)), pixels.argmax(1)] = 1
                            maximum_mean_map[img, conv_filter, r1:r2, c1:c2] = pixels_zeros

                        pooled_imgs[img, conv_filter, row, column] = self.pooling_method(pixels) #mean/max happens here
        return pooled_imgs

    def backward(self, sgd):

        new_sgd = np.zeros(self.input.shape)

       # print sgd.shape  (32, 20, 30, 30)
       # print new_sgd.shape (32, 20, 31, 31)

        num_images = sgd.shape[0]
        pool_num_featureMaps = sgd.shape[1]
        pool_filter_width = sgd.shape[2]
        pool_filter_height = sgd.shape[3]

        for img in range(num_images):
            for conv_filter in range(pool_num_featureMaps):
                for row in range(pool_filter_width):
                    for column in range(pool_filter_height):
                        new_sgd[img, conv_filter, row, column] = sgd[img, conv_filter, row, column] #mean/max happens here

        #print sgd.shape 32,20,30,30
        if self.pooling_method == np.mean:
            # distribute the error over the pixels
            pass
        else:
            # assign the error to the max pixel
            pass

class FullyConnectedLayer(object):
    def __init__(self, prev_stack_size, output_size, weight_decay = 0.01):
        self.activation_fn = "ReLU"
        self.output_size = output_size
        self.prev_stack_size = prev_stack_size

        self.weight_decay = weight_decay

        W_shape = (prev_stack_size, self.output_size)
        self.W = np.random.normal(loc = 0, scale = 0.1, size = W_shape)
        self.b = np.zeros(output_size)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.W) + self.b

    def backward(self, prev_sgd):
        self.old_weights = self.W

        self.W += np.dot(self.input.T, prev_sgd)/prev_sgd.shape[0] - self.weight_decay*self.W
        self.b += np.mean(prev_sgd, axis=0)
        return np.dot(prev_sgd, self.old_weights.T)


class SoftMaxLayer(object):
    def forward(self, input):
        output = np.zeros(input.shape)
        for img in range(input.shape[0]):
            e = np.exp(input[img,] - np.max(input[img,])) # to reduce overflow
            output[img, ] =  e/np.sum(e)

        return output


    def backward(self, real, predicted):
        """
        derivative of 0.5(y-y')^2
        :param real: 32 img x 10 classes (with one class equal 1)
        :param predicted: 32 img x 10 classes (each class has a value between 0 and 1)
        :return: 32 img x 10 derivatives
        """
        return - (real - predicted)


class FlattenLayer(object):
    def forward(self, input):
        self.input = input
        return np.reshape(input, (input.shape[0], -1)) #keep first dim and -1 (flatten) the rest

    def backward(self, prev_sgd): return np.reshape(prev_sgd, self.input.shape)


class DebugLayer(object):
    def forward(self, input):
        import matplotlib.pyplot as plt
        print "input dimensions are " + str(input.shape)
        plt.imshow(input[1,2,])
        plt.show()

    def backward(self, prev_sgd): return prev_sgd #for debugging, pass sgd to next layer

