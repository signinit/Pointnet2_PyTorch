import os
import numpy as np
import torch
import torch.utils.data as data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_data_file(name):
    lines = np.loadtxt(name, delimiter=",")
    return lines[:,0:3], lines[:,3]


class Custom3DSemSeg(data.Dataset):
    def __init__(self, num_points, train=True):
        super().__init__()

        self.train, self.num_points = train, num_points

        data_batchlist, label_batchlist = [], []
        for i in range(91):
            data, label = _load_data_file(os.path.join(BASE_DIR, "city-1/city-block-" + str(i) + ".txt"))
            data_batchlist.append(data)
            label_batchlist.append(label)

        if self.train:
            self.points = data_batchlist[10:]
            self.labels = label_batchlist[10:]
        else:
            self.points = data_batchlist[:10]
            self.labels = label_batchlist[:10]

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, len(self.points))
        np.random.shuffle(pt_idxs)

        current_points = torch.from_numpy(self.points[idx][pt_idxs[:4096]].copy()).float()
        current_labels = torch.from_numpy(self.labels[idx][pt_idxs[:4096]].copy()).long()

        print(current_labels)

        return current_points, current_labels

    def __len__(self):
        return len(self.points)

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass


if __name__ == "__main__":
    dset = Custom3DSemSeg(16, train=True)
    print(dset[0])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    for i, data in enumerate(dloader, 0):
        inputs, labels = data
        if i == len(dloader) - 1:
            print(inputs.size())
