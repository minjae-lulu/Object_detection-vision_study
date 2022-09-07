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
#from google.colab.patches import cv2_imshow
#import utils
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)

class CustomDataset(Dataset): 
    def __init__(self):
        self.root = '/Users/minjaelee/Desktop/mnist_png/' 
        
        self.x_data = []
        self.y_data = []
        
        self.root = self.root + 'training/*/' 
        self.img_path = sorted(glob(self.root + '*.png'))
        
        # print(self.img_path)
        for i in tqdm.tqdm(range(len(self.img_path))):
            img = cv2.imread(self.img_path[i], cv2.IMREAD_UNCHANGED)
            num = self.img_path[i].split("/")[-2]
            self.x_data.append(img)
            self.y_data.append(int(num))
            
        
    def __len__(self): 
        return len(self.img_path)
    
    def __getitem__(self, idx):
        transform1 = torchvision.transforms.ToTensor()
        new_x_data = transform1(self.x_data[idx])
        return new_x_data, self.y_data[idx]

learning_rate = 0.001
training_epochs = 15
batch_size = 100
dataset = CustomDataset()
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# 위치를 반환함
# DATA_PATH_LIST = glob('/Users/minjaelee/Desktop/mnist_png/training/1/*.png')
# print(DATA_PATH_LIST)

# img = cv2.imread('/Users/minjaelee/Desktop/5.png')
# #img = cv2.imread('../6.png')
# aug = iaa.Cutout(nb_iterations=2, size = 0.3) # 개수 2개, 사이즈 0.3
# image_aug = aug(image = img) # 인자 이름을 image로 설정(단일 이미지 적용)
# save_path = '/Users/minjaelee/Desktop/save.png'
# cv2.imwrite(save_path,image_aug)
# print(img.shape)

class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out

    

model = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

total_batch = len(data_loader)

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는레이블.
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))