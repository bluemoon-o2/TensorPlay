"""Small Tensor-list optimizer fast paths.

The backend owns the elementwise loop.  The Python optimizer remains the
source of truth for state initialization and falls back to its reference loop
for unsupported layouts/dtypes or older extensions.
"""

import math

import tensorplay as tp

from ._utils import scalar_value


def foreach(name, *args, **kwargs):
    """Call a generated native foreach overload family by its Torch name."""

    fn = getattr(tp, name, None)
    if fn is None:
        return None
    return fn(*args, **kwargs)


def foreach_available(name):
    return _native(name) is not None


def _native(name):
    """Resolve a generated foreach overload without bypassing its dispatcher."""

    extension = getattr(tp, "_C", None)
    if extension is None or getattr(extension, name, None) is None:
        return None
    return getattr(tp, name, None) or getattr(extension, name, None)


def _call(name, *args, **kwargs):
    fn = _native(name)
    if fn is None:
        return None
    return fn(*args, **kwargs)


def _state_batchable(params, grads, states, steps, *, capturable=False):
    if not _batchable(params, grads, states):
        return False
    if len(steps) != len(params):
        return False
    for param, step in zip(params, steps):
        if (not _is_defined(step) or step.numel() != 1 or
                not step.is_contiguous() or
                (capturable and step.device != param.device)):
            return False
    return True


def _scalar_state_batchable(params, grads, shape_states, scalar_states,
                            *, capturable=False):
    if not _batchable(params, grads, shape_states):
        return False
    if len(scalar_states) != len(params):
        return False
    for param, state in zip(params, scalar_states):
        if (not _is_defined(state) or state.numel() != 1 or
                not state.is_contiguous() or
                (capturable and state.device != param.device)):
            return False
    return True


def _group_lists(*lists):
    """Torch's foreach implementation groups tensor lists by device and dtype."""

    if not lists or not lists[0]:
        return []
    groups = {}
    for index, first in enumerate(lists[0]):
        key = (str(first.device), first.dtype)
        groups.setdefault(key, [[] for _ in lists])
        for group, values in zip(groups[key], lists):
            group.append(values[index])
    return list(groups.values())


def _view_real_lists(*lists):
    return [
        [tp.view_as_real(value) if value.is_complex() else value for value in values]
        for values in lists
    ]


def _is_defined(tensor):
    return getattr(tensor, "_impl_id", 0) != 0


def _batchable(params, grads, state_lists=()):
    if not params or len(params) != len(grads):
        return False
    first = params[0]
    if not (first.is_floating_point() or first.is_complex()):
        return False

    for param, grad in zip(params, grads):
        if getattr(param, "is_sparse", False) or getattr(grad, "is_sparse", False):
            return False
        if (param.dtype != first.dtype or param.device != first.device or
                param.shape != grad.shape or param.dtype != grad.dtype or
                param.device != grad.device or not param.is_contiguous() or
                not grad.is_contiguous()):
            return False
    for state_list in state_lists:
        if len(state_list) != len(params):
            return False
        for param, state in zip(params, state_list):
            if state is None:
                continue
            if not _is_defined(state):
                continue
            if (state.shape != param.shape or state.dtype != param.dtype or
                    state.device != param.device or not state.is_contiguous()):
                return False
    return True


def sgd(params, grads, momentum_buffers, *, lr, momentum, dampening,
        weight_decay, nesterov, first_momentum_step=False):
    fn = getattr(tp._C, "_foreach_sgd", None)
    if params and (params[0].is_complex() or
                   params[0].dtype not in (tp.float32, tp.float64)):
        return False
    state_lists = [momentum_buffers] if momentum else []
    if fn is None or not _batchable(params, grads, state_lists):
        return False
    fn(params=params, grads=grads, momentum_buffers=momentum_buffers,
       lr=scalar_value(lr, "lr"), momentum=scalar_value(momentum, "momentum"),
       dampening=scalar_value(dampening, "dampening"),
       weight_decay=scalar_value(weight_decay, "weight_decay"),
       nesterov=bool(nesterov),
       first_momentum_step=bool(first_momentum_step))
    return True


def sgd_foreach(params, grads, momentum_buffers, *, lr, momentum, dampening,
                weight_decay, nesterov, maximize, first_momentum_step=False):
    """Generic Torch foreach SGD path for dtypes outside the fused kernel."""

    if not params:
        return False
    if not _batchable(params, grads, [momentum_buffers] if momentum else []):
        return False
    required = ["_foreach_add_", "_foreach_add", "_foreach_neg"]
    if momentum:
        required.extend(("_foreach_mul_", "_foreach_copy_"))
    if any(_native(name) is None for name in required):
        return False

    if any(p.is_complex() for p in params):
        params, grads = _view_real_lists(params, grads)
        if momentum:
            (momentum_buffers,) = _view_real_lists(momentum_buffers)
    if maximize:
        grads = _foreach_unary("_foreach_neg", grads)

    if weight_decay:
        if isinstance(weight_decay, tp.Tensor):
            scaled = _foreach_mul(params, weight_decay)
            grads = _foreach_add(grads, scaled)
        elif maximize:
            _foreach_add_(grads, params, alpha=weight_decay)
        else:
            grads = _foreach_add(grads, params, alpha=weight_decay)

    if momentum:
        if first_momentum_step:
            _foreach_copy_(momentum_buffers, grads)
        else:
            _foreach_mul_(momentum_buffers, momentum)
            _foreach_add_(momentum_buffers, grads, alpha=1.0 - dampening)
        if nesterov:
            _foreach_add_(grads, momentum_buffers, alpha=momentum)
            update = grads
        else:
            update = momentum_buffers
    else:
        update = grads

    if isinstance(lr, tp.Tensor):
        scaled = _foreach_mul(update, -lr)
        _foreach_add_(params, scaled)
    else:
        _foreach_add_(params, update, alpha=-scalar_value(lr, "lr"))
    return True


def adam_foreach(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                 steps, *, lr, beta1, beta2, eps, weight_decay, amsgrad,
                 maximize, capturable, decoupled_weight_decay=False):
    """Generic Torch foreach Adam/AdamW path."""

    if not params:
        return False
    states = [exp_avgs, exp_avg_sqs]
    if amsgrad:
        states.append(max_exp_avg_sqs)
    if not _state_batchable(params, grads, states, steps,
                            capturable=capturable):
        return False
    required = (
        "_foreach_add_", "_foreach_add", "_foreach_mul_",
        "_foreach_addcmul_", "_foreach_lerp_", "_foreach_sqrt",
        "_foreach_div_", "_foreach_pow", "_foreach_reciprocal_",
    )
    if amsgrad:
        required += ("_foreach_maximum_",)
    if any(_native(name) is None for name in required):
        return False

    if any(p.is_complex() for p in params):
        params, grads, exp_avgs, exp_avg_sqs = _view_real_lists(
            params, grads, exp_avgs, exp_avg_sqs
        )
        if amsgrad:
            (max_exp_avg_sqs,) = _view_real_lists(max_exp_avg_sqs)
    if maximize:
        grads = _foreach_unary("_foreach_neg", grads)
    _foreach_step_add_(steps)

    if weight_decay:
        if decoupled_weight_decay:
            decay = (1.0 - lr * weight_decay if isinstance(lr, tp.Tensor)
                     else 1.0 - scalar_value(lr, "lr") * weight_decay)
            _foreach_mul_(params, decay)
        elif maximize:
            _foreach_add_(grads, params, alpha=weight_decay)
        else:
            grads = _foreach_add(grads, params, alpha=weight_decay)

    if isinstance(beta1, tp.Tensor):
        # Torch keeps Tensor betas on the parameter device for the capturable
        # path.  ``_foreach_lerp_`` takes a Python Scalar, so express the
        # same recurrence through Tensor multiplication without reading the
        # beta back to the host.
        if beta1.device != params[0].device:
            beta1 = beta1.to(device=params[0].device)
        one_minus_beta1 = tp.ones_like(beta1).sub_(beta1)
        _foreach_mul_(exp_avgs, beta1)
        _foreach_add_(exp_avgs, _foreach_mul(grads, one_minus_beta1))
        beta1_value = None
    else:
        beta1_value = scalar_value(beta1, "beta1")
        _foreach_lerp_(exp_avgs, grads, 1.0 - beta1_value)

    if isinstance(beta2, tp.Tensor):
        if beta2.device != params[0].device:
            beta2 = beta2.to(device=params[0].device)
        one_minus_beta2 = tp.ones_like(beta2).sub_(beta2)
        _foreach_mul_(exp_avg_sqs, beta2)
        scaled_grads = _foreach_mul(grads, one_minus_beta2)
        _foreach_addcmul_(exp_avg_sqs, scaled_grads, grads, value=1.0)
        beta2_value = None
    else:
        beta2_value = scalar_value(beta2, "beta2")
        _foreach_mul_(exp_avg_sqs, beta2_value)
        _foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2_value)

    if amsgrad:
        _call("_foreach_maximum_", max_exp_avg_sqs, exp_avg_sqs)
        exp_avg_sq_sqrt = _foreach_unary("_foreach_sqrt", max_exp_avg_sqs)
    else:
        exp_avg_sq_sqrt = _foreach_unary("_foreach_sqrt", exp_avg_sqs)

    if capturable:
        bias1 = _call("_foreach_pow", beta1, steps)
        _foreach_sub_(bias1, 1.0)
        _foreach_reciprocal_(bias1)
        _foreach_mul_(bias1, lr)
        _foreach_neg_(bias1)

        bias2 = _call("_foreach_pow", beta2, steps)
        _foreach_sub_(bias2, 1.0)
        _foreach_neg_(bias2)
        _call("_foreach_sqrt_", bias2)

        _foreach_div_(exp_avg_sq_sqrt, bias2)
        _foreach_add_(exp_avg_sq_sqrt, eps)
        _foreach_div_(exp_avg_sq_sqrt, bias1)
        _foreach_addcdiv_(params, exp_avgs, exp_avg_sq_sqrt)
    else:
        if beta1_value is None or beta2_value is None:
            raise RuntimeError("Tensor betas require capturable=True")
        step_values = [scalar_value(step, "step") for step in steps]
        step_size = [
            -scalar_value(lr, "lr") / (1.0 - beta1_value ** step)
            for step in step_values
        ]
        bias2 = [
            (1.0 - beta2_value ** step) ** 0.5 for step in step_values
        ]
        _foreach_div_(exp_avg_sq_sqrt, bias2)
        _foreach_add_(exp_avg_sq_sqrt, eps)
        _foreach_addcdiv_(params, exp_avgs, exp_avg_sq_sqrt, step_size)
    return True


def adagrad(params, grads, state_sums, steps, *, lr, lr_decay,
            weight_decay, eps, maximize, differentiable):
    """Torch's multi-tensor Adagrad update.

    The scalar functional in Torch deliberately owns the step increment.  The
    caller therefore only materializes singleton step tensors and passes them
    through unchanged until this function has validated the complete list.
    """

    if differentiable:
        raise RuntimeError("_foreach ops don't support autograd")
    if not params:
        return True
    if not _state_batchable(params, grads, [state_sums], steps):
        return False
    required = (
        "_foreach_add_", "_foreach_addcmul_", "_foreach_sqrt",
        "_foreach_addcdiv_", "_foreach_mul", "_foreach_mul_",
    )
    if any(_native(name) is None for name in required):
        return False

    if any(param.is_complex() for param in params):
        params, grads, state_sums = _view_real_lists(
            params, grads, state_sums
        )
    if maximize:
        grads = _foreach_unary("_foreach_neg", grads)
    _foreach_step_add_(steps)

    if weight_decay:
        if maximize:
            _foreach_add_(grads, params, alpha=weight_decay)
        else:
            grads = _foreach_add(grads, params, alpha=weight_decay)

    lr_value = scalar_value(lr, "lr")
    minus_clr = [
        -lr_value / (1.0 + (scalar_value(step, "step") - 1.0) * lr_decay)
        for step in steps
    ]
    _foreach_addcmul_(state_sums, grads, grads, value=1.0)
    std = _foreach_unary("_foreach_sqrt", state_sums)
    _foreach_add_(std, eps)

    if weight_decay or maximize:
        _foreach_mul_(grads, minus_clr)
        numerator = grads
    else:
        numerator = _foreach_mul(grads, minus_clr)
    _foreach_addcdiv_(params, numerator, std)
    return True


def adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps, *,
         lr, beta1, beta2, eps, weight_decay, amsgrad):
    fn = getattr(tp._C, "_foreach_adam", None)
    if params and (params[0].is_complex() or
                   params[0].dtype not in (tp.float32, tp.float64)):
        return False
    states = [exp_avgs, exp_avg_sqs]
    if amsgrad:
        states.append(max_exp_avg_sqs)
    if fn is None or not _batchable(params, grads, states):
        return False
    fn(params=params, grads=grads, exp_avgs=exp_avgs,
       exp_avg_sqs=exp_avg_sqs, max_exp_avg_sqs=max_exp_avg_sqs,
       steps=steps, lr=scalar_value(lr, "lr"),
       beta1=scalar_value(beta1, "beta1"),
       beta2=scalar_value(beta2, "beta2"), eps=scalar_value(eps, "eps"),
       weight_decay=scalar_value(weight_decay, "weight_decay"),
       amsgrad=bool(amsgrad))
    return True


def _foreach_step_add_(steps):
    if not steps:
        return
    fn = _native("_foreach_add_")
    if fn is not None:
        fn(steps, 1)
        return
    for step in steps:
        step.add_(1.0)


def _foreach_add_(self, other, *, alpha=1):
    if isinstance(other, (list, tuple)):
        return _call("_foreach_add_", self, other, alpha=alpha)
    return _call("_foreach_add_", self, other, alpha=alpha)


def _foreach_add(self, other, *, alpha=1):
    return _call("_foreach_add", self, other, alpha=alpha)


def _foreach_sub_(self, other, *, alpha=1):
    if isinstance(other, (list, tuple)):
        return _call("_foreach_sub_", self, other, alpha=alpha)
    return _call("_foreach_sub_", self, other, alpha=alpha)


def _foreach_sub(self, other, *, alpha=1):
    return _call("_foreach_sub", self, other, alpha=alpha)


def _foreach_mul_(self, other):
    return _call("_foreach_mul_", self, other)


def _foreach_mul(self, other):
    return _call("_foreach_mul", self, other)


def _foreach_div_(self, other):
    return _call("_foreach_div_", self, other)


def _foreach_div(self, other):
    return _call("_foreach_div", self, other)


def _foreach_addcmul_(self, tensor1, tensor2, *, value=1):
    return _call("_foreach_addcmul_", self, tensor1, tensor2, value=value)


def _foreach_addcdiv_(self, tensor1, tensor2, value=1):
    return _call("_foreach_addcdiv_", self, tensor1, tensor2, value=value)


def _foreach_lerp_(self, end, weight):
    return _call("_foreach_lerp_", self, end, weight)


def _foreach_copy_(self, src):
    return _call("_foreach_copy_", self, src)


def _foreach_unary(name, self):
    return _call(name, self)


def _foreach_unary_(name, self):
    return _call(name, self)


def _foreach_neg_(self):
    return _call("_foreach_neg_", self)


def _foreach_reciprocal_(self):
    return _call("_foreach_reciprocal_", self)


def rmsprop(params, grads, square_avgs, grad_avgs, momentum_buffers, steps, *,
            lr, alpha, eps, weight_decay, momentum, centered, maximize,
            capturable, differentiable):
    if differentiable or not params:
        return False
    states = [square_avgs]
    if centered:
        states.append(grad_avgs)
    if momentum:
        states.append(momentum_buffers)
    if not _state_batchable(params, grads, states, steps, capturable=capturable):
        return False
    required = ["_foreach_add_", "_foreach_mul_", "_foreach_addcmul_",
                "_foreach_sqrt", "_foreach_addcdiv_"]
    if centered:
        required.append("_foreach_lerp_")
    if momentum:
        required.append("_foreach_addcdiv_")
    if any(_native(name) is None for name in required):
        return False
    if any(p.is_complex() for p in params):
        params, grads, square_avgs, grad_avgs, momentum_buffers = _view_real_lists(
            params, grads, square_avgs, grad_avgs, momentum_buffers
        )
    _foreach_step_add_(steps)
    if maximize:
        grads = _foreach_unary("_foreach_neg", grads)
    if weight_decay:
        grads = _foreach_add(grads, params, alpha=weight_decay) if not maximize else grads
        if maximize:
            _foreach_add_(grads, params, alpha=weight_decay)
    _foreach_mul_(square_avgs, alpha)
    _foreach_addcmul_(square_avgs, grads, grads, value=1.0 - alpha)
    if centered:
        _foreach_lerp_(grad_avgs, grads, 1.0 - alpha)
        avg = _call("_foreach_addcmul", square_avgs, grad_avgs, grad_avgs, value=-1)
        avg = _foreach_unary("_foreach_sqrt", avg)
    else:
        avg = _foreach_unary("_foreach_sqrt", square_avgs)
    _foreach_add_(avg, eps)
    if momentum:
        _foreach_mul_(momentum_buffers, momentum)
        _foreach_addcdiv_(momentum_buffers, grads, avg)
        if capturable and isinstance(lr, tp.Tensor):
            scaled = _foreach_mul(momentum_buffers, -lr)
            _foreach_add_(params, scaled)
        else:
            _foreach_add_(params, momentum_buffers, alpha=-scalar_value(lr, "lr"))
    elif capturable and isinstance(lr, tp.Tensor):
        scaled = _foreach_mul(grads, -lr)
        _foreach_addcdiv_(params, scaled, avg)
    else:
        _foreach_addcdiv_(params, grads, avg, value=-scalar_value(lr, "lr"))
    return True


def adadelta(params, grads, square_avgs, acc_deltas, steps, *, lr, rho, eps,
             weight_decay, maximize, capturable, differentiable):
    if differentiable or not params:
        return False
    if not _state_batchable(
            params, grads, [square_avgs, acc_deltas], steps,
            capturable=capturable):
        return False
    required = ("_foreach_add_", "_foreach_mul_", "_foreach_addcmul_",
                "_foreach_sqrt", "_foreach_div_")
    if any(_native(name) is None for name in required):
        return False
    if any(p.is_complex() for p in params):
        params, grads, square_avgs, acc_deltas = _view_real_lists(
            params, grads, square_avgs, acc_deltas
        )
    _foreach_step_add_(steps)
    if maximize:
        grads = _foreach_unary("_foreach_neg", grads)
    if weight_decay:
        grads = _foreach_add(grads, params, alpha=weight_decay) if not maximize else grads
        if maximize:
            _foreach_add_(grads, params, alpha=weight_decay)
    _foreach_mul_(square_avgs, rho)
    _foreach_addcmul_(square_avgs, grads, grads, value=1.0 - rho)
    std = _call("_foreach_add", square_avgs, eps)
    _foreach_unary_("_foreach_sqrt_", std)
    deltas = _call("_foreach_add", acc_deltas, eps)
    _foreach_unary_("_foreach_sqrt_", deltas)
    _foreach_div_(deltas, std)
    _foreach_mul_(deltas, grads)
    _foreach_mul_(acc_deltas, rho)
    _foreach_addcmul_(acc_deltas, deltas, deltas, value=1.0 - rho)
    if capturable and isinstance(lr, tp.Tensor):
        _foreach_mul_(deltas, -lr)
        _foreach_add_(params, deltas)
    else:
        _foreach_add_(params, deltas, alpha=-scalar_value(lr, "lr"))
    return True


def adamax(params, grads, exp_avgs, exp_infs, steps, *, lr, beta1, beta2, eps,
           weight_decay, maximize, capturable, differentiable):
    if differentiable or not params:
        return False
    if not _state_batchable(
            params, grads, [exp_avgs, exp_infs], steps,
            capturable=capturable):
        return False
    required = ("_foreach_lerp_", "_foreach_mul_", "_foreach_abs",
                "_foreach_add_", "_foreach_maximum_", "_foreach_addcdiv_")
    if any(_native(name) is None for name in required):
        return False
    if any(p.is_complex() for p in params):
        params, grads, exp_avgs, exp_infs = _view_real_lists(
            params, grads, exp_avgs, exp_infs
        )
    if maximize:
        grads = _foreach_unary("_foreach_neg", grads)
    _foreach_step_add_(steps)
    if weight_decay:
        grads = _foreach_add(grads, params, alpha=weight_decay) if not maximize else grads
        if maximize:
            _foreach_add_(grads, params, alpha=weight_decay)
    _foreach_lerp_(exp_avgs, grads, 1.0 - beta1)
    _foreach_mul_(exp_infs, beta2)
    abs_grads = _foreach_unary("_foreach_abs", grads)
    _foreach_add_(abs_grads, eps)
    _call("_foreach_maximum_", exp_infs, abs_grads)
    if capturable:
        bias = _call("_foreach_pow", beta1, steps)
        _foreach_sub_(bias, 1.0)
        _foreach_div_(bias, lr)
        denom = _call("_foreach_mul", exp_infs, bias)
        _foreach_addcdiv_(params, exp_avgs, denom)
    else:
        values = [scalar_value(lr, "lr") / (1.0 - beta1 ** scalar_value(step, "step")) * -1
                  for step in steps]
        _foreach_addcdiv_(params, exp_avgs, exp_infs, values)
    return True


def asgd(params, grads, axs, mus, etas, steps, *, lr, lambd, t0, alpha,
         weight_decay, maximize, capturable, differentiable):
    """Torch's multi-tensor ASGD update.

    ``mus`` and ``etas`` are singleton tensors rather than Python scalars in
    Torch.  Keeping them in the foreach call is important for CUDA graph
    capture and also avoids a host read on the capturable path.
    """

    if differentiable or not params:
        return False
    if not _state_batchable(params, grads, [axs], steps,
                            capturable=capturable):
        return False
    if not _scalar_state_batchable(params, grads, [], mus,
                                   capturable=capturable):
        return False
    if len(etas) != len(params):
        return False
    for param, eta in zip(params, etas):
        if (not _is_defined(eta) or eta.numel() != 1 or
                not eta.is_contiguous() or
                (capturable and eta.device != param.device)):
            return False
    required = (
        "_foreach_add_", "_foreach_sub", "_foreach_mul_",
        "_foreach_addcmul_", "_foreach_copy_", "_foreach_maximum_",
        "_foreach_reciprocal_", "_foreach_pow_",
    )
    if any(_native(name) is None for name in required):
        return False

    if any(p.is_complex() for p in params):
        params, grads, axs = _view_real_lists(params, grads, axs)
    _foreach_step_add_(steps)
    if maximize:
        grads = _foreach_unary("_foreach_neg", grads)

    if weight_decay:
        if maximize:
            _foreach_add_(grads, params, alpha=weight_decay)
            intermediate = grads
        else:
            intermediate = _foreach_add(grads, params, alpha=weight_decay)
        _foreach_add_(intermediate, params, alpha=lambd)
    else:
        intermediate = _foreach_add(grads, params, alpha=lambd)

    _foreach_addcmul_(params, intermediate, etas, value=-1)
    delta = _foreach_sub(params, axs)
    _foreach_addcmul_(axs, delta, mus)

    if capturable:
        new_mu = _foreach_sub(steps, t0)
        _call("_foreach_maximum_", new_mu, 1.0)
        _call("_foreach_reciprocal_", new_mu)
        _foreach_copy_(mus, new_mu)

        new_eta = _foreach_mul(steps, lambd)
        _foreach_mul_(new_eta, lr)
        _foreach_add_(new_eta, 1.0)
        _call("_foreach_pow_", new_eta, alpha)
        _call("_foreach_reciprocal_", new_eta)
        _foreach_mul_(new_eta, lr)
        _foreach_copy_(etas, new_eta)
    else:
        lr_value = scalar_value(lr, "lr")
        new_etas = [
            tp.tensor(
                lr_value / ((1.0 + lambd * lr_value * scalar_value(step, "step")) ** alpha),
                dtype=eta.dtype,
                device=eta.device,
            )
            for eta, step in zip(etas, steps)
        ]
        new_mus = [
            tp.tensor(
                1.0 / max(1.0, scalar_value(step, "step") - t0),
                dtype=mu.dtype,
                device=mu.device,
            )
            for mu, step in zip(mus, steps)
        ]
        _foreach_copy_(etas, new_etas)
        _foreach_copy_(mus, new_mus)
    return True


def adafactor(params, grads, row_vars, col_vars, variances, steps, *,
              lr, beta2_decay, weight_decay, eps1, eps2, d, maximize):
    """Torch's multi-tensor Adafactor update.

    The reduction-heavy pieces (norms and the two factor products) are still
    one Tensor operation per parameter, exactly as ATen's foreach
    implementation.  Elementwise state and parameter updates go through the
    native foreach families so the Python optimizer does not rebuild the
    algorithm out of scalar Tensor calls.
    """

    if not params:
        return False
    if (len(params) != len(grads) or len(params) != len(row_vars) or
            len(params) != len(col_vars) or len(params) != len(variances) or
            len(params) != len(steps)):
        return False
    if any(p.is_complex() for p in params):
        return False
    if not _state_batchable(params, grads, [], steps):
        return False
    required = (
        "_foreach_add_", "_foreach_mul_", "_foreach_mul",
        "_foreach_lerp_", "_foreach_clamp_min_", "_foreach_rsqrt_",
        "_foreach_neg",
    )
    if any(_native(name) is None for name in required):
        return False

    _foreach_step_add_(steps)
    if maximize:
        grads = _foreach_unary("_foreach_neg", grads)

    lr_value = scalar_value(lr, "lr")
    eps1_value = scalar_value(eps1, "eps1") if eps1 is not None else None
    if eps1_value is None:
        # Torch uses torch.finfo(param.dtype).eps.  Keep the table local so
        # this path remains valid before the dtype-info API is generated.
        eps1_value = {
            tp.float16: 9.765625e-4,
            tp.bfloat16: 7.8125e-3,
            tp.float32: 1.1920928955078125e-7,
            tp.float64: 2.220446049250313e-16,
        }.get(params[0].dtype, 1.1920928955078125e-7)
    step_values = [scalar_value(step, "step") for step in steps]
    one_minus_beta2 = [step ** beta2_decay for step in step_values]
    rho = [min(lr_value, 1.0 / math.sqrt(step)) for step in step_values]
    alphas = [
        max(eps2, float(param.norm(2).item()) / math.sqrt(param.numel())) * rate
        for param, rate in zip(params, rho)
    ]

    if weight_decay:
        _foreach_mul_(params, 1.0 - lr_value * weight_decay)

    matrix_indices = [index for index, grad in enumerate(grads)
                      if grad.ndim > 1]
    vector_indices = [index for index, grad in enumerate(grads)
                      if grad.ndim <= 1]

    if matrix_indices:
        matrix_grads = [grads[index] for index in matrix_indices]
        matrix_rows = [row_vars[index] for index in matrix_indices]
        matrix_cols = [col_vars[index] for index in matrix_indices]
        if any(row is None or col is None
               for row, col in zip(matrix_rows, matrix_cols)):
            return False
        row_means = [
            grad.norm(dim=-1, keepdim=True) for grad in matrix_grads
        ]
        _foreach_mul_(row_means, row_means)
        _call("_foreach_div_", row_means,
              [grad.shape[-1] for grad in matrix_grads])
        _call("_foreach_lerp_", matrix_rows, row_means,
              [one_minus_beta2[index] for index in matrix_indices])

        col_means = [
            grad.norm(dim=-2, keepdim=True) for grad in matrix_grads
        ]
        _foreach_mul_(col_means, col_means)
        _call("_foreach_div_", col_means,
              [grad.shape[-2] for grad in matrix_grads])
        _call("_foreach_lerp_", matrix_cols, col_means,
              [one_minus_beta2[index] for index in matrix_indices])

        var_estimates = [row @ col
                         for row, col in zip(matrix_rows, matrix_cols)]
        row_means = [row.mean(dim=-2, keepdim=True) for row in matrix_rows]
        _call("_foreach_clamp_min_", row_means, eps1)
        _call("_foreach_div_", var_estimates, row_means)
    else:
        var_estimates = []
        matrix_grads = []

    if vector_indices:
        vector_grads = [grads[index] for index in vector_indices]
        vector_variances = [variances[index] for index in vector_indices]
        if any(variance is None for variance in vector_variances):
            return False
        grads_squared = _foreach_mul(vector_grads, vector_grads)
        _call("_foreach_lerp_", vector_variances, grads_squared,
              [one_minus_beta2[index] for index in vector_indices])
        vector_estimates = [variance.clone() for variance in vector_variances]
    else:
        vector_grads = []
        vector_estimates = []

    for estimates, estimate_grads, indices in (
            (var_estimates, matrix_grads, matrix_indices),
            (vector_estimates, vector_grads, vector_indices)):
        if not estimates:
            continue
        _call("_foreach_clamp_min_", estimates, eps1_value * eps1_value)
        _call("_foreach_rsqrt_", estimates)
        _foreach_mul_(estimates, estimate_grads)
        update_scales = []
        for estimate, index in zip(estimates, indices):
            clip = max(
                1.0,
                float(estimate.norm(2).item()) /
                (math.sqrt(estimate.numel()) * d),
            )
            update_scales.append(-alphas[index] / clip)
        _call("_foreach_mul_", estimates, update_scales)
        _call("_foreach_add_", [params[index] for index in indices], estimates)
    return True


def radam(params, grads, exp_avgs, exp_avg_sqs, steps, *, lr, beta1, beta2,
          eps, weight_decay, decoupled_weight_decay, maximize, capturable,
          differentiable):
    """Torch's multi-tensor RAdam update."""

    if differentiable or not params:
        return False
    if not _state_batchable(params, grads, [exp_avgs, exp_avg_sqs], steps,
                            capturable=capturable):
        return False
    required = (
        "_foreach_add_", "_foreach_mul_", "_foreach_addcmul_",
        "_foreach_lerp_", "_foreach_sqrt", "_foreach_div_",
        "_foreach_reciprocal_", "_foreach_pow", "_foreach_neg_",
    )
    if any(_native(name) is None for name in required):
        return False

    if any(p.is_complex() for p in params):
        params, grads, exp_avgs, exp_avg_sqs = _view_real_lists(
            params, grads, exp_avgs, exp_avg_sqs
        )
    _foreach_step_add_(steps)
    if maximize:
        grads = _foreach_unary("_foreach_neg", grads)

    rho_inf = 2.0 / (1.0 - beta2) - 1.0
    if capturable:
        beta2_pow = _call("_foreach_pow", beta2, steps)
        bias1 = _foreach_unary("_foreach_neg", beta2_pow)
        _foreach_add_(bias1, 1.0)
        rho_t = _foreach_mul(steps, beta2_pow)
        _foreach_mul_(rho_t, 2.0)
        _foreach_div_(rho_t, bias1)
        _foreach_neg_(rho_t)
        _foreach_add_(rho_t, rho_inf)
    else:
        step_values = [scalar_value(step, "step") for step in steps]
        beta2_pows = [beta2 ** step for step in step_values]
        rho_t = [
            rho_inf - 2.0 * step * power / (1.0 - power)
            for step, power in zip(step_values, beta2_pows)
        ]

    if weight_decay:
        if decoupled_weight_decay:
            decay = 1.0 - lr * weight_decay if isinstance(lr, tp.Tensor) else 1.0 - scalar_value(lr, "lr") * weight_decay
            _foreach_mul_(params, decay)
        elif maximize:
            _foreach_add_(grads, params, alpha=weight_decay)
        else:
            grads = _foreach_add(grads, params, alpha=weight_decay)

    _foreach_lerp_(exp_avgs, grads, 1.0 - beta1)
    _foreach_mul_(exp_avg_sqs, beta2)
    _foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

    if capturable:
        num = _foreach_sub(rho_t, 4.0)
        sub2 = _foreach_sub(rho_t, 2.0)
        _foreach_mul_(num, sub2)
        _foreach_mul_(num, rho_inf)
        denom = _foreach_mul(rho_t, (rho_inf - 4.0) * (rho_inf - 2.0))
        _foreach_div_(num, denom)
        _call("_foreach_sqrt_", num)
        # ``num`` is the square-rooted rectification factor; Torch's branch
        # is selected by rho_t itself, not by that numerator.
        rect = [tp.where(rho_value > 5.0, value, 0.0)
                for rho_value, value in zip(rho_t, num)]
        unrect_step_size = [tp.where(value > 0.0, 0.0, 1.0) for value in rect]
        _foreach_mul_(unrect_step_size, lr)

        bias_correction1 = _call("_foreach_pow", beta1, steps)
        _foreach_neg_(bias_correction1)
        _foreach_add_(bias_correction1, 1.0)
        _foreach_div_(unrect_step_size, bias_correction1)
        _foreach_neg_(unrect_step_size)

        bias_correction2 = _call("_foreach_pow", beta2, steps)
        _foreach_neg_(bias_correction2)
        _foreach_add_(bias_correction2, 1.0)
        _call("_foreach_sqrt_", bias_correction2)
        _foreach_mul_(bias_correction2, lr)
        _foreach_mul_(bias_correction2, rect)
        _foreach_neg_(bias_correction2)
        _foreach_div_(bias_correction2, bias_correction1)
    else:
        rect = [
            ((value - 4.0) * (value - 2.0) * rho_inf /
             ((rho_inf - 4.0) * (rho_inf - 2.0) * value)) ** 0.5
            if value > 5.0 else 0.0
            for value in rho_t
        ]
        unrectified = [0.0 if value > 0.0 else 1.0 for value in rect]
        bias1_values = [1.0 - beta1 ** scalar_value(step, "step") for step in steps]
        unrect_step_size = [
            -scalar_value(lr, "lr") * value / correction
            for value, correction in zip(unrectified, bias1_values)
        ]
        bias_correction2 = [
            ((1.0 - beta2 ** scalar_value(step, "step")) ** 0.5)
            * scalar_value(lr, "lr") * value / correction * -1.0
            for step, value, correction in zip(steps, rect, bias1_values)
        ]

    buffer = _foreach_unary("_foreach_sqrt", exp_avg_sqs)
    _foreach_add_(buffer, eps)
    _foreach_div_(buffer, bias_correction2)
    _call("_foreach_reciprocal_", buffer)
    _foreach_add_(buffer, unrect_step_size)
    _foreach_addcmul_(params, exp_avgs, buffer)
    return True


def nadam(params, grads, exp_avgs, exp_avg_sqs, mu_products, steps, *,
          beta1, beta2, lr, weight_decay, momentum_decay, eps,
          decoupled_weight_decay, maximize, capturable, differentiable):
    """Torch's multi-tensor NAdam update."""

    if differentiable or not params:
        return False
    if not _state_batchable(params, grads, [exp_avgs, exp_avg_sqs], steps,
                            capturable=capturable):
        return False
    if len(mu_products) != len(params):
        return False
    for param, value in zip(params, mu_products):
        if (not _is_defined(value) or value.numel() != 1 or
                not value.is_contiguous() or
                (capturable and value.device != param.device)):
            return False
    required = (
        "_foreach_add_", "_foreach_mul_", "_foreach_addcmul_",
        "_foreach_lerp_", "_foreach_sqrt", "_foreach_pow",
        "_foreach_neg_", "_foreach_sub", "_foreach_div_",
    )
    if any(_native(name) is None for name in required):
        return False

    if any(p.is_complex() for p in params):
        params, grads, exp_avgs, exp_avg_sqs = _view_real_lists(
            params, grads, exp_avgs, exp_avg_sqs
        )
    _foreach_step_add_(steps)
    if maximize:
        grads = _foreach_unary("_foreach_neg", grads)

    if weight_decay:
        if decoupled_weight_decay:
            decay = (1.0 - lr * weight_decay if isinstance(lr, tp.Tensor)
                     else 1.0 - scalar_value(lr, "lr") * weight_decay)
            _foreach_mul_(params, decay)
        elif maximize:
            _foreach_add_(grads, params, alpha=weight_decay)
        else:
            grads = _foreach_add(grads, params, alpha=weight_decay)

    _foreach_lerp_(exp_avgs, grads, 1.0 - beta1)
    _foreach_mul_(exp_avg_sqs, beta2)
    _foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)
    exp_avg_sq_sqrt = _foreach_unary("_foreach_sqrt", exp_avg_sqs)

    if capturable:
        exponent = _foreach_mul(steps, momentum_decay)
        mus = _call("_foreach_pow", 0.96, exponent)
        _foreach_mul_(mus, -0.5)
        _foreach_add_(mus, 1.0)
        _foreach_mul_(mus, beta1)
        _foreach_add_(exponent, momentum_decay)
        mu_nexts = _call("_foreach_pow", 0.96, exponent)
        _foreach_mul_(mu_nexts, -0.5)
        _foreach_add_(mu_nexts, 1.0)
        _foreach_mul_(mu_nexts, beta1)
        bias_correction_sqrt = _call("_foreach_pow", beta2, steps)
        _foreach_sub_(bias_correction_sqrt, 1.0)
        _foreach_neg_(bias_correction_sqrt)
        _call("_foreach_sqrt_", bias_correction_sqrt)
    else:
        step_values = [scalar_value(step, "step") for step in steps]
        bias_correction_sqrt = [
            (1.0 - beta2 ** step) ** 0.5 for step in step_values
        ]
        mus = [
            beta1 * (1.0 - 0.5 * (0.96 ** (step * momentum_decay)))
            for step in step_values
        ]
        mu_nexts = [
            beta1 * (1.0 - 0.5 * (0.96 ** ((step + 1.0) * momentum_decay)))
            for step in step_values
        ]

    _foreach_mul_(mu_products, mus)
    _foreach_div_(exp_avg_sq_sqrt, bias_correction_sqrt)
    _foreach_add_(exp_avg_sq_sqrt, eps)

    if capturable:
        _foreach_sub_(mus, 1.0)
        _foreach_mul_(mus, lr)
        denom = _foreach_sub(mu_products, 1.0)
        _foreach_neg_(denom)
        _foreach_div_(mus, denom)

        denom = _foreach_mul(mu_products, mu_nexts)
        _foreach_mul_(mu_nexts, lr)
        _foreach_sub_(denom, 1.0)
        _foreach_div_(mu_nexts, denom)

        numerator = _foreach_mul(mus, grads)
        _foreach_addcmul_(numerator, mu_nexts, exp_avgs)
        _foreach_addcdiv_(params, numerator, exp_avg_sq_sqrt)
    else:
        lr_value = scalar_value(lr, "lr")
        step_size_grads = [
            -lr_value * (1.0 - mu) /
            (1.0 - scalar_value(product, "mu_product"))
            for product, mu in zip(mu_products, mus)
        ]
        step_size_expavg = [
            -lr_value * mu_next /
            (1.0 - scalar_value(product, "mu_product") * mu_next)
            for product, mu_next in zip(mu_products, mu_nexts)
        ]
        _foreach_addcdiv_(params, grads, exp_avg_sq_sqrt, step_size_grads)
        _foreach_addcdiv_(params, exp_avgs, exp_avg_sq_sqrt, step_size_expavg)
    return True


def rprop(params, grads, prevs, step_sizes, steps, *, step_size_min,
          step_size_max, etaminus, etaplus, maximize, capturable,
          differentiable):
    """Torch's multi-tensor Rprop update."""

    if differentiable or not params:
        return False
    if not _state_batchable(params, grads, [prevs, step_sizes], steps,
                            capturable=capturable):
        return False
    required = (
        "_foreach_add_", "_foreach_mul", "_foreach_mul_", "_foreach_copy_",
        "_foreach_sign_", "_foreach_sign", "_foreach_addcmul_",
        "_foreach_clamp_min_", "_foreach_clamp_max_",
    )
    if any(_native(name) is None for name in required):
        return False

    if any(p.is_complex() for p in params):
        params, grads, prevs, step_sizes = _view_real_lists(
            params, grads, prevs, step_sizes
        )
    _foreach_step_add_(steps)
    signs = _foreach_mul(grads, prevs)
    if maximize:
        _foreach_neg_(signs)

    _foreach_copy_(prevs, grads)
    if maximize:
        _foreach_neg_(prevs)
    _call("_foreach_sign_", signs)
    for sign in signs:
        sign.copy_(tp.where(sign > 0, etaplus,
                            tp.where(sign < 0, etaminus, 1.0)))
    _foreach_mul_(step_sizes, signs)
    _call("_foreach_clamp_min_", step_sizes, step_size_min)
    _call("_foreach_clamp_max_", step_sizes, step_size_max)

    updated_grads = prevs
    for grad, sign in zip(updated_grads, signs):
        grad.copy_(tp.where(sign == etaminus, 0.0, grad))
    grad_signs = _foreach_unary("_foreach_sign", updated_grads)
    _foreach_addcmul_(params, grad_signs, step_sizes, value=-1)
    return True
