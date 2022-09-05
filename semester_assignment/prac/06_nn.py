import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = nn.Linear(1,1) # input dim =1, output dim = 1 w 1개, b 1개 세팅됨, 만약 다중이면 3,1 하고 w3개 b1개 된다
print(list(model.parameters())) # 출력해보면 일단 w,b 랜덤 초기화 되어서 나온다.

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(201):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train) # (hypho - y_train)**2 랑 동일하다
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, 200, cost.item()
      ))
        
print("\n\n Let's test!!")
new_var = torch.FloatTensor([[4.0]])
pred_y = model(new_var)
print("pred y val when x = 4", pred_y)
print("parameters after learn ", list(model.parameters()))