import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


x = torch.tensor([1, 2])
y = torch.tensor([3, 4, 5])
z = torch.tensor([6, 7, 8, 9])
grid_x, grid_y, grid_z = torch.meshgrid(x,y,z)

print(grid_x.shape)
print("grid_x result: ", grid_x)

print(grid_y.shape)
print("grid_y result: ", grid_y)