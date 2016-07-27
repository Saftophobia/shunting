from scipy.fftpack import convolve

__author__ = 'saftophobia'
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

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

def draw_image(arr):
    #plt.imshow(arr.transpose((1,2,0)))
    plt.imshow(arr)
    plt.show()

