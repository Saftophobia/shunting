__author__ = 'saftophobia'

import matplotlib.pyplot as plt
import numpy as np
import time, logging

def draw_image(arr):
    reshaped = np.reshape(arr, (3, 32, 32))
    plt.imshow(reshaped.transpose((1,2,0)))
    plt.show()

def shuffle_data(dataset):
    seed = int(time.time())
    logging.info("shuffling data with seed: %d" % seed)

    r = np.random.RandomState(seed)
    r.shuffle(dataset.data)
    r = np.random.RandomState(seed)
    r.shuffle(dataset.labels)



