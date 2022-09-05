import torch
import torch.nn as nn
import torch.nn.functional as F


## 반드시 숙지해라 모델의 기본 class 구조이다. 

# init과 forward는 이미 nn.Module에 있어서 overide하는 것이다.
class MultivariateLinearRegressionModel(nn.Module): # nn.Module을 상속받는 함수
    def __init__(self):
        super().__init__() # 부모 클래스인 nn.Module 을 super()을 사용하여 초기화 시켜줘야 한다
        self.linear = nn.Linear(3,1)
        
    def forward(self,x):
        return self.linear(x)


x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = MultivariateLinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2001):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {}'.format(
          epoch, 2000, cost.item()
      ))