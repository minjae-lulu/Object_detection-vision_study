import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("current device is: ", device)

# 대표적인 nerf data로 포크레인 여러 방향과 각도에서 찍은 사진이 있다. 
raw_data = np.load("tiny_nerf_data.npz")

images = raw_data["images"]
poses = raw_data["poses"]
focal = raw_data["focal"]

print("raw_data: ", raw_data)
print("image: ",images.shape)
print("poses: ",poses.shape)
print("focal: ",focal)

H, W = images.shape[1:3] # 두번째 세번째 h,w를 사용
H = int(H)
W = int(W)
print("H and W is: ", H, W)

testimg, testpose = images[99], poses[99]

# plt.imshow(testimg)
# plt.show()

images = torch.Tensor(images).to(device)
poses = torch.Tensor(poses).to(device)
testimg = torch.Tensor(testimg).to(device)
testpose = torch.Tensor(testpose).to(device)


def get_rays(H, W, focal, pose):
    i, j = torch.meshgrid(
        torch.arange(W, dtype = torch.float32),
        torch.arange(H, dtype = torch.float32)
    )
    i = i.t()
    j = j.t()
    dirs = torch.stack(
        [(i - W*0.5)/focal,
         -(j-H*0.5)/focal,
         -torch.ones_like(i)], -1).to(device)
    
    rays_d = torch.sum(dirs[..., np.newaxis, :] * pose[:3, :3], -1)
    rays_o = pose[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

ray_o, ray_d = get_rays(H, W, focal, testpose)
print("rays: ", ray_o.shape, ray_d.shape)


def positional_encoder(x, L_embed=6):
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.**i *x))
    return torch.cat(rets, -1)

def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    cumprod = torch.cumprod(tensor, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.
    return cumprod