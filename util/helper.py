from scipy.fftpack import convolve

__author__ = 'saftophobia'
import numpy as np
import scipy.ndimage

def L1(params): return np.sum(np.abs(params))
def L2(params): return np.sum(params ** 2)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_diff(x): return sigmoid(x) * (1 - sigmoid(x))
def reLU_soft(x): return np.log(1 + np.exp(x)) #log = ln
def reLU_soft_diff(x): return sigmoid(x)
def reLU(x): return np.maximum(0.0, x)
def reLU_d(x):
    dx = np.zeros(x.shape)
    dx[x > 0] = 1
    return dx

def convolute(input_img, num_of_output_featureMaps, filter_shape, W, b, activation_function):
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

    feature_maps = np.zeros((num_input_pooled_imgs, num_of_output_featureMaps, input_img_width, input_img_height))

    for img in range(num_input_pooled_imgs):
        for featureMap_num in range(num_of_output_featureMaps):
            feature_map = np.zeros((input_img_width, input_img_height))

            for input_num in range(prev_layer_stack_size):
                image = input_img[img, input_num, : , :]
                filter = W[input_num, featureMap_num, ] #filter for each pooled image in the stack

                feature_map = feature_map + scipy.ndimage.filters.convolve(image, filter, mode='reflect')

            feature_map_w_bias = activation_function(feature_map) + b[featureMap_num]
            feature_maps[img, featureMap_num, :, : ] = feature_map_w_bias

    return feature_maps


def subsample(input_feature_maps, pool_size, strides=1, padding=0, pooling_method=np.max):
    """ Returns pooled images from convolutional filters
    :param input_feature_maps: output of previous convolutional layers e.g 32 img x 16 featureMap x 32 width x 32 height
    :param pool_size: pool size e.g (2,2)
    :param strides: how many steps the filter should take in every iteration
    :param padding: start from inner frame of the image #NOT_IMPLEMENTED
    :param method: pooling methid is either PoolingMethods.MEAN or PoolingMethods.MAX
    :return pooled_imgs: pooled images e.g 32img x 6 layers x 16x16
    """
    num_images = input_feature_maps.shape[0]
    convoluted_num_featureMaps = input_feature_maps.shape[1]
    convoluted_filter_width = input_feature_maps.shape[2]
    convoluted_filter_height = input_feature_maps.shape[3]


    pooled_img_width  = ((convoluted_filter_width + 2 * padding - pool_size)/strides) + 1
    pooled_img_height = ((convoluted_filter_height + 2 * padding - pool_size)/strides) + 1
    if not (float(pooled_img_width).is_integer() and float(pooled_img_height).is_integer()):
        raise Exception("Invalid HyperParameters for subsample Layer! pooled image width/height is %.4f" % pooled_img_width)

    pooled_imgs = np.zeros((num_images, convoluted_num_featureMaps, pooled_img_width, pooled_img_height))

    for img in range(num_images):
        for conv_filter in range(convoluted_num_featureMaps):
            for row in range(pooled_img_width):
                r1 = row * pool_size/2
                r2 = r1  + pool_size/2
                for column in range(pooled_img_height):
                    c1 = column * pool_size/2 * strides
                    c2 = c1 + pool_size/2
                    pixels = input_feature_maps[img, conv_filter, r1:r2, c1:c2]
                    pooled_imgs[img, conv_filter, row, column] = pooling_method(pixels)
    return pooled_imgs
