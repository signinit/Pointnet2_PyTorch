import os
import numpy as np
import torch
import torch.utils.data as data

from .CustomClassificationLoader import Custom3DClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_data_file(name):
    lines = np.loadtxt(name, delimiter=",")
    return lines[:,0:3]


class Custom3DLinear(Custom3DClassification):
    def __init__(self, batch_dir, batch_file, num_points, train=True):
        super().__init__()

        self.train, self.num_points = train, num_points

        file = open(os.path.join(BASE_DIR, batch_dir, batch_file), 'r')
        batches = file.read().splitlines()
        batch_amount = len(batches)
        training_percentage = 0.1
        training_amount = round(training_percentage * batch_amount)

        if(training_amount == 0):
            raise ValueError("to few data, training amount is 0")

        data_batchlist, label_batchlist = [], []
        for batch in batches:
            batch_url, label = batch.split(";")
            data = _load_data_file(os.path.join(BASE_DIR, batch_dir, batch_url))
            data_batchlist.append(data)
            label_batchlist.append(label)

        if self.train:
            self.points = np.array(data_batchlist[training_amount:])
            self.labels = np.array(label_batchlist[training_amount:])
        else:
            self.points = np.array(data_batchlist[:training_amount])
            self.labels = np.array(label_batchlist[:training_amount])
