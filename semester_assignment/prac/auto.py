import cv2
import torch

x = torch.ones(2,2,requires_grad = True) # 미분할수있게 x에 대한 연산들을 추적하게 한다.
print(x)

y = x + 1
print(y)

z = -1*y**3 
print(z)

res = z.mean()
print(res)

res.backward() # 미분하여 최적화
print(x.grad)