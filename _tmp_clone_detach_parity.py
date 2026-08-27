"""Parity suite: clone / detach / detach_ / _is_view vs system torch (CPU).

Each case builds the SAME base data (numpy) into torch and tp, applies the
same view op on both, then compares clone/detach behavior.
"""
import numpy as np
import torch
import tensorplay as tp

FAIL = []
OK = 0

def check(name, cond, detail=""):
    global OK
    if cond:
        OK += 1
    else:
        FAIL.append(f"{name} {detail}")

def leaf(t):
    v = t.is_leaf
    return v() if callable(v) else v

def same_vals(a, b):
    return np.allclose(a.detach().cpu().numpy(), b.detach().cpu().numpy())

def pair(base_np, op):
    """Apply op to torch and tp tensors built from the same data."""
    tt = op(torch.from_numpy(base_np).clone())
    pt = op(tp.tensor(base_np))
    return tt, pt

RNG = np.random.default_rng(0)
def rnd(*shape):
    return RNG.standard_normal(shape).astype(np.float32)

# ---------------------------------------------------------------- clone -----
cases = [
    ("transpose", lambda t: t.t(), (4, 6)),
    ("permute3d", lambda t: t.permute(2, 0, 1), (2, 3, 4)),
    ("size1-dim", lambda t: t.reshape(1, 3).t(), (3,)),
    ("1d-slice", lambda t: t[::3], (10,)),
    ("sliced-nondense", lambda t: t[:, ::2], (4, 6)),
    ("expand-overlap", lambda t: t.reshape(2, 1).expand(2, 3), (2, 1)),
    ("contiguous", lambda t: t, (3, 4)),
    ("select-view", lambda t: t[1], (4, 6)),
]
for name, op, shape in cases:
    base = rnd(*shape)
    tt, pt = pair(base, op)
    tc, pc = tt.clone(), pt.clone()
    check(f"clone[{name}].shape", list(tc.shape) == list(pc.shape),
          f"torch={list(tc.shape)} tp={list(pc.shape)}")
    check(f"clone[{name}].stride", list(tc.stride()) == list(pc.stride()),
          f"torch={list(tc.stride())} tp={list(pc.stride())}")
    check(f"clone[{name}].values", same_vals(tc, pc))
    check(f"clone[{name}].offset", tc.storage_offset() == pc.storage_offset(),
          f"torch={tc.storage_offset()} tp={pc.storage_offset()}")

# 0-dim / empty
tc, pc = torch.tensor(3.0).clone(), tp.tensor(3.0).clone()
check("clone[0dim]", list(tc.shape) == list(pc.shape) and same_vals(tc, pc))
for shape in [(2, 0), (0, 3)]:
    tc = torch.empty(*shape).clone(); pc = tp.empty(*shape).clone()
    check(f"clone[empty{shape}].stride", list(tc.stride()) == list(pc.stride()),
          f"torch={list(tc.stride())} tp={list(pc.stride())}")

# clone never aliases, version 0
x = torch.randn(4); xt = tp.tensor(x.numpy())
yc, ptc = x.clone(), xt.clone()
yc[0] = 99; ptc[0] = 99
check("clone.no-alias", x[0].item() != 99 and xt[0].item() != 99)
check("clone.version0", x.clone()._version == 0 and xt.clone()._version == 0)

# memory_format on 4D/5D
m4 = rnd(2, 3, 4, 5); p4 = tp.tensor(m4)
for fmt_t, fmt_p, fname in [
    (torch.channels_last, tp.channels_last, "channels_last"),
    (torch.contiguous_format, tp.contiguous_format, "contiguous_format"),
    (torch.preserve_format, tp.preserve_format, "preserve_format"),
]:
    tc = torch.from_numpy(m4).clone(memory_format=fmt_t)
    pc = p4.clone(memory_format=fmt_p)
    check(f"clone.mf[{fname}].stride", list(tc.stride()) == list(pc.stride()),
          f"torch={list(tc.stride())} tp={list(pc.stride())}")
    check(f"clone.mf[{fname}].values", same_vals(tc, pc))
m5 = rnd(2, 3, 4, 5, 6)
tc = torch.from_numpy(m5).clone(memory_format=torch.channels_last_3d)
pc = tp.tensor(m5).clone(memory_format=tp.channels_last_3d)
check("clone.mf[channels_last_3d].stride", list(tc.stride()) == list(pc.stride()),
      f"torch={list(tc.stride())} tp={list(pc.stride())}")

# transposed + explicit contiguous / preserve
base = rnd(2, 3)
tt = torch.from_numpy(base).clone().t()
pt = tp.tensor(base).t()
check("clone.mf[contig-on-transposed]",
      list(tt.clone(memory_format=torch.contiguous_format).stride()) ==
      list(pt.clone(memory_format=tp.contiguous_format).stride()))
check("clone.mf[preserve-on-transposed]",
      list(tt.clone(memory_format=torch.preserve_format).stride()) ==
      list(pt.clone(memory_format=tp.preserve_format).stride()))
check("clone.mf[preserve-on-transposed].values",
      same_vals(tt.clone(memory_format=torch.preserve_format),
                pt.clone(memory_format=tp.preserve_format)))

# channels-last input cloned with preserve keeps CL strides
mcl = torch.from_numpy(m4).contiguous(memory_format=torch.channels_last)
pcl = p4.contiguous(memory_format=tp.channels_last)
check("clone.preserve-CL", list(mcl.clone().stride()) == list(pcl.clone().stride()),
      f"torch={list(mcl.clone().stride())} tp={list(pcl.clone().stride())}")
check("clone.preserve-CL.values", same_vals(mcl.clone(), pcl.clone()))

# rank errors (exact messages)
for shape, fmt_t, fmt_p, fname in [
    ((2, 3), torch.channels_last, tp.channels_last, "cl-rank2"),
    ((2, 3, 4), torch.channels_last, tp.channels_last, "cl-rank3"),
    ((2, 3, 4, 5, 6), torch.channels_last, tp.channels_last, "cl-rank5"),
    ((2, 3, 4, 5), torch.channels_last_3d, tp.channels_last_3d, "cl3d-rank4"),
]:
    te = pe = None
    try:
        torch.randn(*shape).clone(memory_format=fmt_t)
    except RuntimeError as e:
        te = str(e)
    try:
        tp.randn(*shape).clone(memory_format=fmt_p)
    except RuntimeError as e:
        pe = str(e)
    check(f"clone.mf-err[{fname}]", te is not None and te == pe,
          f"torch={te!r} tp={pe!r}")

# autograd
a_np = rnd(3)
a = torch.from_numpy(a_np).clone().requires_grad_(True)
at = tp.tensor(a_np, requires_grad=True)
ca, cpt = a.clone(), at.clone()
check("clone.requires_grad", ca.requires_grad and cpt.requires_grad)
check("clone.grad_fn", "CloneBackward" in type(ca.grad_fn).__name__ and
      "CloneBackward" in cpt.grad_fn.name,
      f"torch={type(ca.grad_fn).__name__} tp={cpt.grad_fn.name}")
ca.sum().backward(); cpt.sum().backward()
check("clone.grad", same_vals(a.grad, at.grad))

# ---------------------------------------------------------------- detach ----
x_np = rnd(3)
x = torch.from_numpy(x_np).clone(); xt = tp.tensor(x_np)
d, dt = x.detach(), xt.detach()
check("detach.requires_grad", d.requires_grad == dt.requires_grad == False)
check("detach.grad_fn", d.grad_fn is None and dt.grad_fn is None)
check("detach.is_leaf", leaf(d) and leaf(dt))
check("detach.data_ptr", d.data_ptr() == x.data_ptr() and dt.data_ptr() == xt.data_ptr())
x.add_(1); xt.add_(1)
check("detach.version-shared", x._version == d._version and xt._version == dt._version,
      f"torch x={x._version} d={d._version}; tp x={xt._version} d={dt._version}")
check("detach.values", same_vals(x, xt))

# detach of grad-requiring non-leaf
a = torch.from_numpy(x_np).clone().requires_grad_(True); b = a * 2
at = tp.tensor(x_np, requires_grad=True); bt = at * 2
db, dbt = b.detach(), bt.detach()
check("detach.nonleaf.requires_grad", db.requires_grad == dbt.requires_grad == False)
check("detach.nonleaf.grad_fn", db.grad_fn is None and dbt.grad_fn is None)

# detach of a view is fine, and the result is NOT a view
v = torch.from_numpy(rnd(4, 4)).t(); vt = tp.randn(4, 4).t()
dv, dvt = v.detach(), vt.detach()
check("detach.of-view._is_view", dv._is_view() == dvt._is_view() == False)
check("detach.of-view.data", dv.data_ptr() == v.data_ptr() and dvt.data_ptr() == vt.data_ptr())

# functional detach
fd = torch.detach(x); fdt = tp.detach(xt)
check("detach.functional", fd.data_ptr() == x.data_ptr() and fdt.data_ptr() == xt.data_ptr())

# ---------------------------------------------------------------- _is_view --
view_cases = [
    ("t", lambda t: t.t(), (4, 4)),
    ("transpose", lambda t: t.transpose(0, 2), (2, 3, 4)),
    ("permute", lambda t: t.permute(1, 2, 0), (2, 3, 4)),
    ("select", lambda t: t[2], (4, 5)),
    ("slice", lambda t: t[1:3], (4, 5)),
    ("slice-step", lambda t: t[::2], (8,)),
    ("expand", lambda t: t.reshape(1, 3).expand(4, 3), (3,)),
    ("squeeze", lambda t: t.reshape(1, 3, 1).squeeze(), (3,)),
    ("unsqueeze", lambda t: t.reshape(3).unsqueeze(0), (3,)),
    ("view", lambda t: t.reshape(4, 3).view(12), (12,)),
    ("flatten", lambda t: t.flatten(), (2, 3, 4)),
    ("movedim", lambda t: t.movedim(0, 2), (2, 3, 4)),
    ("diagonal", lambda t: t.reshape(4, 4).diagonal(), (4, 4)),
]
for name, op, shape in view_cases:
    base = rnd(*shape)
    tv, pv = pair(base, op)
    check(f"_is_view[{name}]", tv._is_view() == pv._is_view() == True,
          f"torch={tv._is_view()} tp={pv._is_view()}")

# reshape: alias -> view, copy -> not view
ra = torch.from_numpy(rnd(4, 3)).reshape(12); pa = tp.randn(4, 3).reshape(12)
check("_is_view[reshape-alias]", ra._is_view() == pa._is_view() == True)
rc = torch.from_numpy(rnd(4, 3)).t().reshape(12); pc = tp.randn(4, 3).t().reshape(12)
check("_is_view[reshape-copy]", rc._is_view() == pc._is_view() == False)

# non-views
nv_cases = [
    ("clone", lambda t: t.t().clone(), (4, 4)),
    ("contiguous-copy", lambda t: t.t().contiguous(), (4, 4)),
    ("fresh", lambda t: t + 0, (3,)),
]
for name, op, shape in nv_cases:
    base = rnd(*shape)
    tv, pv = pair(base, op)
    check(f"_is_view[{name}]==False", tv._is_view() == pv._is_view() == False,
          f"torch={tv._is_view()} tp={pv._is_view()}")
base = rnd(4)
tv = torch.from_numpy(base).index_select(0, torch.tensor([1, 2]))
pv = tp.tensor(base).index_select(0, tp.tensor([1, 2]))
check("_is_view[index_select]==False", tv._is_view() == pv._is_view() == False,
      f"torch={tv._is_view()} tp={pv._is_view()}")

# contiguous() on an already-contiguous view returns the same view
u = torch.from_numpy(rnd(4)).unsqueeze(0); ut = tp.randn(4).unsqueeze(0)
check("_is_view[contiguous-noop]", u.contiguous()._is_view() == ut.contiguous()._is_view() == True)

# views under no_grad are still views
with torch.no_grad():
    tn = torch.randn(4, 4, requires_grad=True).t()
check("_is_view[no_grad]", tn._is_view() == True)

# index_select: copy semantics -> fresh version counter
x = torch.from_numpy(rnd(4)).clone(); xt = tp.tensor(x.numpy())
y = x.index_select(0, torch.tensor([1, 2])); yt = xt.index_select(0, tp.tensor([1, 2]))
x.add_(1); xt.add_(1)
check("index_select.version", x._version == 1 and y._version == 0 and
      xt._version == 1 and yt._version == 0,
      f"torch y={y._version} tp yt={yt._version}")

# view(dtype): torch does NOT mark it as a view; version counter IS shared
vd = torch.from_numpy(rnd(4)).view(torch.int32)
vdt = tp.tensor(rnd(4)).view(tp.int32)
check("_is_view[view-dtype]", vd._is_view() == vdt._is_view() == False,
      f"torch={vd._is_view()} tp={vdt._is_view()}")
xs = torch.from_numpy(rnd(4)).clone(); xst = tp.tensor(xs.numpy())
sv, svt = xs.view(torch.float32), xst.view(tp.float32)
check("view-dtype.same._is_view", sv._is_view() == svt._is_view() == False)
check("view-dtype.same.data", sv.data_ptr() == xs.data_ptr() and svt.data_ptr() == xst.data_ptr())
xs.add_(1); xst.add_(1)
check("view-dtype.version-shared", xs._version == sv._version and xst._version == svt._version)
# detach_ on view(dtype) result is legal (not a view)
try:
    torch.from_numpy(rnd(4)).view(torch.int32).detach_()
    t_ok = True
except RuntimeError:
    t_ok = False
try:
    tp.tensor(rnd(4)).view(tp.int32).detach_()
    p_ok = True
except RuntimeError:
    p_ok = False
check("view-dtype.detach_-legal", t_ok and p_ok)

# ---------------------------------------------------------------- detach_ ---
msg_t = msg_p = None
try:
    torch.randn(4, 4).t().detach_()
except RuntimeError as e:
    msg_t = str(e)
try:
    tp.randn(4, 4).t().detach_()
except RuntimeError as e:
    msg_p = str(e)
check("detach_.view-error", msg_t is not None and msg_t == msg_p,
      f"torch={msg_t!r} tp={msg_p!r}")

try:
    torch.randn(4, 4)[1:].detach_()
    t_ok = False
except RuntimeError:
    t_ok = True
try:
    tp.randn(4, 4)[1:].detach_()
    p_ok = False
except RuntimeError:
    p_ok = True
check("detach_.slice-view-error", t_ok and p_ok)

with torch.no_grad():
    w = torch.randn(4, 4, requires_grad=True).t()
try:
    w.detach_()
    t_ok = False
except RuntimeError:
    t_ok = True
try:
    tp.randn(4, 4).t().detach_()
    p_ok = False
except RuntimeError:
    p_ok = True
check("detach_.no_grad-view-error", t_ok and p_ok)

# on a leaf: clears requires_grad, returns self
w = torch.randn(3, requires_grad=True); wt = tp.tensor(w.detach().numpy(), requires_grad=True)
r = w.detach_(); rt = wt.detach_()
check("detach_.leaf.requires_grad", w.requires_grad == wt.requires_grad == False)
check("detach_.leaf.same-obj", r is w and rt is wt)
check("detach_.leaf.is_leaf", leaf(w) and leaf(wt))

# on a non-leaf: clears grad_fn, becomes leaf
z = torch.randn(3, requires_grad=True) * 2
zt = tp.tensor(z.detach().numpy(), requires_grad=True) * 2
z.detach_(); zt.detach_()
check("detach_.nonleaf.grad_fn", z.grad_fn is None and zt.grad_fn is None)
check("detach_.nonleaf.requires_grad", z.requires_grad == zt.requires_grad == False)
check("detach_.nonleaf.is_leaf", leaf(z) and leaf(zt))

# ---------------------------------------------------------------------------
print(f"PASS {OK}")
if FAIL:
    print(f"FAIL {len(FAIL)}")
    for f in FAIL:
        print("  -", f)
    raise SystemExit(1)
print("ALL-OK")
