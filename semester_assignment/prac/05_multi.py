import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])


print(x_train.shape)
print(y_train.shape)

W = torch.zeros((3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W,b], lr= 1e-5)

for epoch in range(201):
    
    hypothesis = x_train.matmul(W) + b # b는 브로드캐스팅되어 각 셈플에 더해진다.
    cost = torch.mean((hypothesis - y_train)**2)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    print('Epoch {:4d}/200 hypo: {} cost: {:.6f}'.format (epoch,  hypothesis.squeeze().detach(), cost.item()))
    # hypothesis.squeeze().detach()를 써서 표현한다.
    
    
