import torch
import torch.nn as nn

# 배치 크기 × 채널 × 높이(height) × 너비(widht)의 크기의 텐서를 선언
inputs = torch.Tensor(1, 1, 28, 28)

# 1채널 짜리를 입력받아서 32채널을 뽑아내는데 커널(filter) 사이즈는 3이고 패딩은 1.
conv1 = nn.Conv2d(1,32,3, padding=1)
#print(conv1)

conv2 = nn.Conv2d(32,64,3, padding=1)
pool = nn.MaxPool2d(2)

out = conv1(inputs)
out = pool(out)
out = conv2(out)
out = pool(out)
out = out.view(out.size(0),-1) # 배치차원 빼고 전부 flatten

#fc layer 로 10개로 information 을 줄여준다. 
fc = nn.Linear(3136, 10) #input 3136, output 10
out = fc(out)
print(out.shape)
