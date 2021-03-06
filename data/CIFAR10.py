__author__ = 'saftophobia'

import os, cPickle, logging
import numpy as np
from time import time

class CIFAR10:
    TAGS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, data_batches_count, dir = "/data/cifar-10-batches-py"):
        logging.info("Loading %d data batches" % data_batches_count)
        self.data   = []
        self.labels = []

        for i in range(1, data_batches_count + 1):
            logging.info("Loading data batch %d" % i)

            f = open(os.getcwd() + dir + "/data_batch_%d" %i, 'rb') #singlebatch
            dict = cPickle.load(f)
            f.close()

            self.labels.extend(dict["labels"]) #singledim(1,10k)
            for img in dict["data"]:
                self.data.append(self.array_to_3RGB(img)) #singledim(10k,3,1024)

        self.data   = np.array(self.data)
        self.labels = np.array(self.labels)

        self.shuffle_data(self.data, self.labels)

        # reshape array to a nxn matrix
        img_dim = int(np.sqrt(self.data.shape[2]))
        self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], img_dim, img_dim)


    def array_to_3RGB(self, array):
        return np.array(array).reshape((3,1024))

    def shuffle_data(self, data, labels):
        seed = int(time())
        logging.info("shuffling data with seed: %d" % seed)

        r = np.random.RandomState(seed)
        r.shuffle(data)
        r = np.random.RandomState(seed)
        r.shuffle(labels)

