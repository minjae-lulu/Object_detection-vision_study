import cv2
import torch

# tt = torch.ones(3,2)

# print(torch.empty(3,2))
# print(torch.ones(3,3))
# print(torch.zeros(2))
# print(torch.rand(2,3))
# print(torch.tensor([13,4,2])) #리스트나 nparray도 tensor로 변환가능
# print(tt.size())
# print(type(tt))

# x = torch.rand(2,2)
# y = torch.rand(2,2)
# print(x)
# print(y)

# # print(x+y)
# # print(torch.add(x,y))
# print(y.add(x)) # 얘는 y = x+y를 의미한다 아래도 마찬가지
# print(y.add_(x))
# print(y[1,1])
# print(y[:,1])

# z = torch.rand(6,6)
# print(z)

# print(z.view(36))
# print(z.view(12,3)) # resize tensor.

# k = z.numpy() #ndarray로 변환
# print(k)

ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)

ft = ft.squeeze()
print(ft) # squeeze를 통해 2차원이였는데, 1차원으로 줄어들었다. 단 차원이 1인경우만 차원줄이기 가능
print(ft.shape)

ft = ft.unsqueeze(0) # 첫번째에 1차원을 추가한다.
print(ft)
print(ft.shape)