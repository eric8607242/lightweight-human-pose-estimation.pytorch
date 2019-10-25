from os import listdir
from os.path import isfile, join

import numpy as np
import PIL

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config import CONFIG

class ToTensor(object):
    def __call__(self, sample):
        source_img, target_img = sample["source"], sample["target"]

        source_img = source_img.transpose((2, 0, 1))
        target_img = target_img.transpose((2, 0, 1))

        return {
                "source" : torch.from_numpy(source_img).float(),
                "target" : torch.from_numpy(target_img).float()
                }


class ADDADataset(Dataset):
    def __init__(self):
        self.input_size = CONFIG["dataset"]["input_size"]

        self.source_img_folder = CONFIG["dataset"]["source_img_folder"]
        self.target_img_folder = CONFIG["dataset"]["target_img_folder"]

        self.source_paths = np.array([self._image_processing(join(self.source_img_folder, f)) for f in listdir(self.source_img_folder) if isfile(join(self.source_img_folder, f))])
        self.target_paths = np.array([self._image_processing(join(self.target_img_folder, f)) for f in listdir(self.target_img_folder) if isfile(join(self.target_img_folder, f))])

        self.transform = transforms.Compose([ToTensor()])

    def _image_processing(self, path):
        img = PIL.Image.open(path)

        img = img.convert("RGB")
        img = img.resize((self.input_size, self.input_size), PIL.Image.ANTIALIAS)

        return np.array(img)

    def __len__(self):
        return len(self.target_paths)


    def __getitem__(self, idx):
        source_img = self.source_paths[idx]
        target_img = self.target_paths[idx]

        data = {"source" : source_img, "target" : target_img}

        if self.transform:
            data = self.transform(data)

        return data
        


