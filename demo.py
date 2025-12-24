import tensorplay

x = tensorplay.ones((4, 4), device="cuda")
print(x.shape)
