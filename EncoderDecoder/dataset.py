import cv2
import math
import numpy as np
from pathlib import Path
import sys
import torch

class GE_Dataset(torch.utils.data.Dataset):
    def data_to_tensor(self, data):
        return torch.from_numpy(data.transpose(2, 0, 1)).float()

    def normalize_label(self, label):
        # {altitude}_{distance}_{fov}_{heading}_{tilt}_{rotation}
        label[0] = str(float(label[0]) / 1000)
        label[1] = str(float(label[1]) / 1000)
        label[2] = str(math.radians(float(label[2])))
        label[3] = str(math.radians(float(label[3])))
        label[4] = str(math.radians(float(label[4])))
        label[5] = str(math.radians(float(label[5])))
        return label

    def filename_to_label(self, filename):
        label = filename.split('_')
        label = self.normalize_label(label)
        label.pop(2) # remove fov from list
        # label.pop(1) # remove distance from list
        # label.pop(0) # remove altitude from list
        label = list(map(float, label))
        return torch.from_numpy(np.asarray(label))

    def is_overhead_view(self, filename):
        """
        Will check last 3 label values to see if image is an overhead view.
        Note: there are multiple overhead views per dataset that differ only by altitude. Current logic will set first
        encounterd overhead view as the 'overhead_view' for the entire dataset.
        """
        label = filename.split('_')
        if float(label[5]) == float(label[4]) == float(label[3]) == 0:
            return True
        return False

    def __init__(self, path):
        self.items = []
        self.overhead_view = None

        if not path.is_dir():
            sys.exit(f'Directory "{path}" does not exist')

        for filename in path.iterdir():
            img = cv2.imread(str(filename))
            x = (self.data_to_tensor(img) / 255.0).cuda()
            if self.overhead_view is None and self.is_overhead_view(filename.stem):
                self.overhead_view = x
            y = self.filename_to_label(filename.stem).cuda()
            self.items.append((x, y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
            

# test = GE_Dataset(Path("E:\Research\code\data\Brooklyn Bridge"))
# print(len(test))
