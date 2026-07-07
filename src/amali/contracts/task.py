import torch

x = torch.rand(5, 2)
y = torch.rand(2,2)
print(x)
print(y)
z= torch.full((2, 2), 2)
d = torch.rand(2, 2)
y.add_(d)
print(y)
print(d, z.dtype)

tensors2 = torch.full((3, 3), 0.234)
zedis = torch.rand(5, 2)
print(tensors2)
crypton = torch.rand(3, 2, 2, dtype=torch.float16)
print(crypton)
zd = x*zedis
zd = torch.mul(x, zedis)
print(zd)
lexa = torch.rand(3, 4, dtype=torch.float32)
print(lexa)
dc = lexa * x
dc = torch.mul(lexa, x)
print(dc)
