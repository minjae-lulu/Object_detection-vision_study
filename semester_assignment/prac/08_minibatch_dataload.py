import torch
import torch.nn as nn
import torch.nn.functional as F

# 미니 배치 쓰는이유: 전체로 loss 계산하면 너무 오래걸리고, 1개로 하면 불안정해서, 미니배치단위로 loss 최적화
# 주로 2^n으로 잡는데, cpu gpu 메모리 2^n이라 효율적 송수신 가능

from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])


dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# 데이터 로더는 2개 인자가 기본이고, 주로 추가로 suffle true를 많이 사용
# batch_size를 적게 할수록 많이 쪼갠다 바꾸면서 출력해보면 알것! (2^n으로)

model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 


nb_epochs = 20
for epoch in range(nb_epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        print(batch_idx)
        print(samples)
        
        x_train, y_train = samples
        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))