import tensorplay as tp

x = tp.zeros([2, 3])
print("type(x):", type(x))
print("mro:", [c.__name__ for c in type(x).__mro__])
print("has dynamic dict:", hasattr(x, "__dict__"))
print("module:", type(x).__module__)

for name in ["add", "mul", "sum", "reshape"]:
    fn = getattr(type(x), name, None)
    print(name, "->", fn)

try:
    r = x.add(1.0)
    print("x.add(1.0) OK:", r.sum().item())
except Exception as e:
    print("x.add failed:", type(e).__name__, e)

try:
    y = tp.add(x, x)
    print("tp.add(x,x) OK:", y.sum().item())
except Exception as e:
    print("tp.add failed:", type(e).__name__, e)
