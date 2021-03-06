import os
import numpy as np
import torch
import torch.utils.data as data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_data_file(name):
    lines = np.loadtxt(name, delimiter=",")
    return lines[:,0:3]


class Custom3DClassification(data.Dataset):
    def __init__(self, batch_dir, batch_file, num_points, transforms=None, train=True):
        super().__init__()
        
        self.transforms = transforms

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
            batch_url, cls = batch.split(";")
            data = _load_data_file(os.path.join(BASE_DIR, batch_dir, batch_url))
            data_batchlist.append(data)
            label_batchlist.append(int(cls))

        if self.train:
            self.points = np.array(data_batchlist[training_amount:])
            self.labels = np.array(label_batchlist[training_amount:])
        else:
            self.points = np.array(data_batchlist[:training_amount])
            self.labels = np.array(label_batchlist[:training_amount])


    def __getitem__(self, idx):
        all_pt_idxs = np.arange(0, self.points[idx].shape[0])
        np.random.shuffle(all_pt_idxs)
        pt_idxs = all_pt_idxs[:self.num_points]

        current_points = torch.from_numpy(self.points[idx, pt_idxs]).float()
        
        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points, self.labels[idx]

    def __len__(self):
        return len(self.points)

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass


if __name__ == "__main__":
    dset = Custom3DClassification("", "", 16, train=True)
    print(dset[0])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    for i, data in enumerate(dloader, 0):
        inputs, labels = data
        if i == len(dloader) - 1:
            print(inputs.size())
