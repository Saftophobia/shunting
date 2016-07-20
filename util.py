__author__ = 'saftophobia'

import matplotlib.pyplot as plt
import numpy as np

def draw_image(matrix):
    plt.matshow(matrix, fignum=100, cmap=plt.cm.gray)
    plt.show()

def draw_image_from_list(list):
    side = int(np.sqrt(list.size))
    draw_image(np.reshape(list, (side, side)))

def shuffle_data(list):
    np.random.shuffle(list)


