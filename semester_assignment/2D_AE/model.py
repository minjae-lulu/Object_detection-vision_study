import torch
import torch.nn as nn
import utils

# Encode - input : 256*256*3 size image, output : 100*1*1 feature
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            #state (3*256*256)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            #state (16*128*128)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            #state (32*64*64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            #state (64*32*32)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            #state (128*16*16)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State (256x8x8)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x4x4)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=1, padding=0)
            # output of main module --> State (1024x1x1)   
            
        )
        self.last_layer = nn.Sequential(
            nn.Linear(1024, latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )
        self.fc_mu = nn.Linear(in_features = 1024, out_features = latent_dim)
        self.fc_logvar = nn.Linear(in_features = 1024, out_features = latent_dim)
        
    def forward(self, img):
        features = self.model(img)
        features = features.view(img.shape[0],-1)
        x_mu = self.fc_mu(features)
        x_logvar = self.fc_logvar(features)
        return x_mu, x_logvar
        
        
# Decoder input : 100*1*1 image features, output : 256*256*3 images        
class Decoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        #...
        print('ddd')
        
        
        
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        