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

        if(training_amount == 0):
            raise ValueError("to few data, training amount is 0")

        data_batchlist, label_batchlist = [], []
        for batch_url in batches:
            data, label = _load_data_file(os.path.join(BASE_DIR, batch_dir, batch_url))
            data_batchlist.append(data)
            label_batchlist.append(label)

        if self.train:
            self.points = np.array(data_batchlist[training_amount:])
            self.labels = np.array(label_batchlist[training_amount:])
        else:
            self.points = np.array(data_batchlist[:training_amount])
            self.labels = np.array(label_batchlist[:training_amount])


    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        print(self.num_points)
        print(pt_idxs[0:10])
        np.random.shuffle(pt_idxs)

        current_points = torch.from_numpy(self.points[idx, pt_idxs]).float()
        current_labels = torch.from_numpy(self.labels[idx, pt_idxs]).long()

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
