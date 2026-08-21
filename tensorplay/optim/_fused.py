"""Torch-shaped wrappers for the native fused optimizer entry points.

The algorithms are implemented by the CPU/CUDA kernels registered under the
same operator names as ATen.  This module only performs the state-list
bookkeeping that Torch's functional optimizer helpers do around those calls.
"""

import tensorplay as tp


def _native(name):
    # The Python functional wrapper can exist in a source checkout before the
    # extension has been regenerated.  Treat that state as unavailable rather
    # than entering the wrapper and failing halfway through optimizer.step().
    if getattr(tp._C, name, None) is None:
        return None
    return getattr(tp, name, None) or getattr(tp._C, name, None)


def _increment_steps(steps):
    if not steps:
        return
    foreach_add = _native("_foreach_add_")
    if foreach_add is not None:
        foreach_add(steps, 1)
    else:
        # This is still graph-capturable: add_ is a native Tensor operation;
        # it only exists as a compatibility fallback for an older extension.
        for step in steps:
            step.add_(1.0)


def _rollback_steps(steps, found_inf):
    if found_inf is None or not steps:
        return
    foreach_sub = _native("_foreach_sub_")
    if foreach_sub is not None:
        foreach_sub(steps, [found_inf] * len(steps))
    else:
        for step in steps:
            step.sub_(found_inf)


def sgd(params, grads, momentum_buffers, *, lr, momentum, dampening,
        weight_decay, nesterov, maximize, is_first_step,
        grad_scale=None, found_inf=None):
    fn = _native("_fused_sgd_")
    if fn is None:
        return False
    fn(
        params,
        grads,
        momentum_buffers,
        weight_decay,
        momentum,
        lr,
        dampening,
        bool(nesterov),
        bool(maximize),
        bool(is_first_step),
        grad_scale,
        found_inf,
    )
    return True


def adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
         state_steps, *, lr, beta1, beta2, weight_decay, eps, amsgrad,
         maximize, decoupled_weight_decay=False, grad_scale=None,
         found_inf=None):
    fn = _native("_fused_adamw_" if decoupled_weight_decay else "_fused_adam_")
    if fn is None:
        return False
    _increment_steps(state_steps)
    try:
        fn(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            bool(amsgrad),
            bool(maximize),
            grad_scale,
            found_inf,
        )
    except Exception:
        # Keep the state transition atomic when argument/device validation in
        # the native entry point fails before a kernel is launched.
        raise
    if found_inf is not None:
        _rollback_steps(state_steps, found_inf)
    return True


def adagrad(params, grads, state_sums, state_steps, *, lr, lr_decay,
            weight_decay, eps, maximize, grad_scale=None, found_inf=None):
    fn = _native("_fused_adagrad_")
    if fn is None:
        return False
    _increment_steps(state_steps)
    try:
        fn(
            params,
            grads,
            state_sums,
            state_steps,
            lr,
            lr_decay,
            weight_decay,
            eps,
            bool(maximize),
            grad_scale,
            found_inf,
        )
    except Exception:
        raise
    if found_inf is not None:
        _rollback_steps(state_steps, found_inf)
    return True
