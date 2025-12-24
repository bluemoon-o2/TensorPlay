import tensorplay as tp

x = tp.tensor([1, 2, 3]).cuda()
y = tp.tensor([4, 5, 6], device=x.device)
print(type(x))
print(y)
print(x + y)
