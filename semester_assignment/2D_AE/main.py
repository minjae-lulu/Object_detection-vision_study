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
from dataloader import CustomDataset
from test import Tester
from train import Trainer
import utils

if __name__ == '__main__':
    epochs = 10
    batchSize = 1
    learningRate = 1e-4

    trainer = Trainer(epochs, batchSize, learningRate)
    trainer.train()

    # tester = Tester(batchSize)
    # tester.test()
