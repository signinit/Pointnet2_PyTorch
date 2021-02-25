import os
import numpy as np
import torch
import torch.utils.data as data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_data_file(name):
    lines = np.loadtxt(name, delimiter=",")
    return lines[:,0:3], lines[:,3]


class Custom3DSemSeg(data.Dataset):
    def __init__(self, batch_dir, batch_file, num_points, train=True):
        super().__init__()

        self.train, self.num_points = train, num_points

        file = open(os.path.join(BASE_DIR, batch_dir, batch_file), 'r')
        batches = file.read().splitlines()
        batch_amount = len(batches)
        training_percentage = 0.1
        training_amount = round(training_percentage * batch_amount)

        data_batchlist, label_batchlist = [], []
        for batch_url in batches:
            data, label = _load_data_file(os.path.join(BASE_DIR, batch_dir, batch_url))
            data_batchlist.append(data)
            label_batchlist.append(label)

        if self.train:
            self.points = data_batchlist[training_amount:]
            self.labels = label_batchlist[training_amount:]
        else:
            self.points = data_batchlist[:training_amount]
            self.labels = label_batchlist[:training_amount]
        
        print(self.points[0].shape)
        print(self.points[0].shape)

    def __getitem__(self, idx):
        pt_idxs = np.arange(self.points[idx].shape[0])
        np.random.shuffle(pt_idxs)


        indices = pt_idxs[:4096]

        current_points = torch.from_numpy(self.points[idx][indices].copy()).float()
        current_labels = torch.from_numpy(self.labels[idx][indices].copy()).long()

        return current_points, current_labels

    def __len__(self):
        return len(self.points)

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass


if __name__ == "__main__":
    dset = Custom3DSemSeg("", "", 16, train=True)
    print(dset[0])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    for i, data in enumerate(dloader, 0):
        inputs, labels = data
        if i == len(dloader) - 1:
            print(inputs.size())
