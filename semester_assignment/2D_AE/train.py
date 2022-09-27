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

class Trainer(object):
    def __init__(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self._build_model()
        self.binary_cross_entropy = torch.nn.BCELoss()

        dataset = CustomDataset(method = 'train')
        self.root = dataset.root
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate, betas=(0.9, 0.999))

        print("Training...")

    

    def _build_model(self):
        net = VAE()
        self.net = net.to(device)
        self.net.train()

    def vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = self.binary_cross_entropy(recon_x.view(-1, 256*256*3), x.view(-1, 256*256*3))
        kldivergence = -0.5 * torch.sum(1+ logvar - mu.pow(2) - logvar.exp())
        #return recon_loss + 0.00001 * kldivergence  # more close average distribution
        return recon_loss + 0.0000001 * kldivergence # more close original

    def train(self):
        for epoch in tqdm.tqdm(range(self.epochs + 1)):
            if epoch % 50 == 0:
                torch.save(self.net.state_dict(), "_".join(
                    ['/home/minjaelee/Desktop/coding/Vision_code/semester_assignment/2D_AE/model', str(epoch),'.pth']))
                    #['/Users/minjaelee/Desktop/coding/Vision_code/semester_assignment/2D_AE/model', str(epoch),'.pth']))

            for batch_idx, samples in enumerate(self.dataloader):
                x_train, y_train = samples
                x_train, y_train = x_train.to(device), y_train.to(device)

                # vae reconstruction
                image_batch_recon, latent_mu, latent_logvar = self.net(x_train)

                # reconstruction error
                loss = self.vae_loss(image_batch_recon, x_train, latent_mu, latent_logvar)

                # backpropagation
                self.optimizer.zero_grad()
                loss.backward()

                # one step of the optimizer (using the gradients from backpropagation)
                self.optimizer.step()

            print('Epoch [%d / %d] vae loss error: %f' % (epoch + 1, self.epochs, loss))
        print('Finish training!')


if __name__ == '__main__':
    print('test')