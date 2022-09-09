import torch
import numpy as np
import torchvision
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import torch.nn.functional as F
from glob import glob
import pandas

class CustomDataset(Dataset):
    def __init__(self, method= None):
        self.root = '/Users/minjaelee/Desktop/mnist_png/'
        self.x_data = []
        self.y_data = []

        if method == 'train':
            self.root = self.root + 'training/*/'
        elif method == 'test':
            self.root = self.root + 'testing/*/'

        self.image_path = sorted(glob(self.root + '*.png'))

        for i in tqdm.tqdm(range(len(self.image_path))):
            img = cv2.imread(self.image_path[i], cv2.IMREAD_UNCHANGED)
            # = self.image_path[i].split("/")[-2]
            img = cv2.resize(img, dsize=(256, 256))
            self.x_data.append(img)
            self.y_data.append(img)

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        transform1 = torchvision.transforms.ToTensor()
        new_x_data = transform1(self.x_data[idx])
        return new_x_data, self.y_data[idx]


if __name__ == '__main__':
    dataset = CustomDataset()
    print("test")

