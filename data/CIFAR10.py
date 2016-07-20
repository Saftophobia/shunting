__author__ = 'saftophobia'

import os, cPickle, logging
import numpy as np

class CIFAR10:
    def __init__(self, dir = "cifar-10-batches-py"):
        self.train  = []
        self.test   = []

        


