import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#torch.manual_seed(1) # 랜덤시드값 고정

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])
W = torch.ones(1, requires_grad=True) # 값이 변경되는 변수, 추적한다 의미
b = torch.ones(1, requires_grad=True)

optimizer = optim.SGD([W,b], lr= 0.01) # W,b를 최적화 하겠다


epochs = 2000
for epoch in range(epochs +1):
    
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train) ** 2)   
    
    optimizer.zero_grad() # grad 초기화, 파이토치는 미분값 누적시키는 특징있어 발산방지
    cost.backward() # cost 미분해 grad 계산
    optimizer.step() # W,b 업데이트
    
    if epoch % 100 == 0:
        print('Epoch {:4d}/{}  W:{:.3f}, b:{:.3f}, cost:{:.3f}'.format(epoch,epochs, W.item(),b.item(),cost.item()))

