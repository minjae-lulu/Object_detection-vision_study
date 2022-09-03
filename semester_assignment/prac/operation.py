import torch

# t = torch.FloatTensor([[3,5,7,8,1,3],[2,3,4,1,2,5]])
# print(t)

# print(t.dim())
# print(t.shape)
# print(t.size())

# print(t[:,1])
# print(t[:,1].size())
# print(t[:,:-1],'\n')

# #broadcasting 자동으로 되서 조심해야함
# m1 = torch.FloatTensor([[1,2]])
# m2 = torch.FloatTensor([3])
# print(m1+m2, '\n') # 3 이 각각 요소에 더해진다

# m1 = torch.FloatTensor([[1,2]])
# m2 = torch.FloatTensor([[3],[4]])
# print(m1+m2, '\n') # 3,4 이 각각 요소에 더해진다 차원이 2*2가 된다

# m1 = torch.FloatTensor([[1, 2], [3, 4]])
# m2 = torch.FloatTensor([[1], [2]])
# print(m1.matmul(m2), '\n') # 행렬곱

# m1 = torch.FloatTensor([[1, 2], [3, 4]])
# m2 = torch.FloatTensor([[1], [2]])
# print(m1 * m2) 
# print(m1.mul(m2), '\n') # 단순 broadcasting 곱


# t = torch.FloatTensor([[1, 2], [3, 4]])
# print(t.max()) # 4 리턴
# print(t.max(dim=0)) # 3,4 가 리턴되지만 argmax(3과 4의 인덱스)리턴함

x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
print(torch.cat([x,y])) # 연결시켜준다. same with dim=0 -> 0번째 차원을 늘려라
print(torch.cat([x,y], dim=1)) # 1번째 차원(2번쨰요소)을 늘려라


x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x,y,z])) #3개이상은 stack가능
