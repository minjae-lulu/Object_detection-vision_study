import torch
import torch.nn.functional as F


z = torch.FloatTensor([1, 2, 3])
hypothesis = F.softmax(z, dim=0)

a = torch.FloatTensor([[2],[1]])
print(F.softmax(a, dim=0))