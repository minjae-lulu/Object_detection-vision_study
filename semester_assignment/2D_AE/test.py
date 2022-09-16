from model import VAE
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
import utils
from dataloader import CustomDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)

class Tester(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self._build_model()

        dataset = CustomDataset(method = 'test')
        self.root = dataset.root
        self.dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = False)
        self.datalen = dataset.__len__()
        self.mse_all_img = []

        # Load of pretrained_wight
        weight_PATH = '/Users/minjaelee/Desktop/coding/Vision_code/semester_assignment/2D_AE/model_10_.pth'
        self.net.load_state_dict(torch.load(weight_PATH))

        print('Testing')

    def _build_model(self):
        net = VAE()
        self.net = net.to(device)
        self.net.eval()
        print('Finish build model.')

    def test(self):
        for batch_idx, samples in enumerate(self.dataloader):
            if batch_idx > 10:
                continue

            x_test, y_test = samples
            out = self.net(x_test.to(device))
            x_test2 = 256. * x_test
            out2 = 256. * out[0]

            basic_path = '/Users/minjaelee/Desktop/coding/Vision_code/semester_assignment/2D_AE/result/'

            # permute로 가로 세로 맞게 쌓아주는 역할임

            abnomal = utils.compare_images_colab(x_test2[0].clone().permute(1, 2, 0).cpu().detach().numpy(), out2[0].clone().permute(1, 2, 0).cpu().detach().numpy(), None, 0.2)
            cv2.imwrite((basic_path + 'test_%d_ori.png') % batch_idx,
                        x_test2[0].clone().permute(1, 2, 0).cpu().detach().numpy())
            cv2.imwrite((basic_path + 'test_%d_gen.png') % batch_idx,
                        out2[0].clone().permute(1, 2, 0).cpu().detach().numpy())
            cv2.imwrite((basic_path + 'test_%d_diff.png') % batch_idx, abnomal)

        print('Finish testing!!')
