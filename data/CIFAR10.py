__author__ = 'saftophobia'

import os, cPickle, logging
import numpy as np

class CIFAR10:

    def __init__(self, data_batches_count, dir = "/data/cifar-10-batches-py"):
        logging.info("Loading %d data batches ..." % data_batches_count)
        self.data   = []
        self.labels = []

        for i in range(1, data_batches_count + 1):
            f = open(os.getcwd() + dir + "/data_batch_%d" %i, 'rb') #singlebatch
            dict = cPickle.load(f)
            f.close()
            self.data.extend(dict["data"]) #singledim(10k,3072)
            self.labels.extend(dict["labels"]) #singledim(1,10k)

        self.data   = np.array(self.data)
        self.labels = np.array(self.labels)

