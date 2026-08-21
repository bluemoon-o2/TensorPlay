import yaml
import os
import re
import argparse
import copy

try:
    from .yaml_utils import YamlLoader
except ImportError:
    try:
        from yaml_utils import YamlLoader
    except ImportError:
        YamlLoader = yaml.SafeLoader # Fallback

try:
    from . import codegen_utils
except ImportError:
    import codegen_utils

# Type mapping from YAML schema to C++ signature
TYPE_MAP = {
    'int64_t[]': 'const std::vector<int64_t>&',
    'str': 'std::string',
    'Tensor[]': 'const std::vector<Tensor>&',
    # The alias annotation describes mutation of the Tensor elements, not of
    # the list container.  Passing the small handle vector by value keeps the
    # generated pybind ABI usable for Python lists while preserving the
    # storage/version mutation of every element.
    'Tensor(a!)[]': 'std::vector<Tensor>',
    'Scalar[]': 'const std::vector<Scalar>&',
    'DType': 'DType',
    'Device': 'Device',
    'double': 'double',
    # ATen uses ``float`` for optimizer hyperparameters.  Keep the schema
    # spelling while using C++ double for the public ABI, matching the
    # backend's opmath convention and Python's float representation.
    'float': 'double',
    'bool': 'bool',
    'Scalar': 'Scalar',
    'Scalar?': 'std::optional<Scalar>',
    'Tensor': 'const Tensor&',
    'Tensor(a!)': 'Tensor&',
    'Scalar?': 'std::optional<Scalar>',
    'int64_t?': 'std::optional<int64_t>',
    'Tensor?': 'const std::optional<Tensor>&',
    'Device?': 'std::optional<Device>',
}

# Type mapping for DispatchStub template arguments
STUB_TYPE_MAP = {
    'int64_t[]': 'const std::vector<int64_t>&',
    'str': 'std::string',
    'Tensor[]': 'const std::vector<Tensor>&',
    'Tensor(a!)[]': 'std::vector<Tensor>',
    'Scalar[]': 'const std::vector<Scalar>&',
    'DType': 'DType',
    'Device': 'Device',
    'double': 'double',
    'float': 'double',
    'bool': 'bool',
    'Scalar': 'Scalar',
    'Scalar?': 'std::optional<Scalar>',
    'Tensor': 'const Tensor&',
    'Tensor(a!)': 'Tensor&',
    'Scalar?': 'std::optional<Scalar>',
    'int64_t?': 'std::optional<int64_t>',
    'Tensor?': 'std::optional<Tensor>',
    'Device?': 'std::optional<Device>',
}

# Autocast op policies, mirroring aten/src/ATen/autocast_mode.cpp
# (AT_FORALL_LOWER_PRECISION_FP / AT_FORALL_FP32 / AT_FORALL_FP32_SET_OPT_DTYPE
# / AT_FORALL_PROMOTE and the CPU-only KERNEL_CPU lists), intersected with the
# operators available in native_functions.yaml.  Each entry is registered as a
# kernel on the AutocastCPU/AutocastCUDA dispatch keys; the generated dispatch
# sites consult those keys before the autograd keys so casts are
# autograd-exposed and inputs are saved for backward post-cast.
AUTOCAST_POLICY = {
    # lower_precision_fp: cast all (eligible) inputs to the autocast dtype.
    'lower_precision_fp': [
        'mm', 'matmul', 'addmm', 'bmm', 'baddbmm',
        'conv1d', 'conv2d', 'conv3d',
        'conv_transpose2d', 'conv_transpose3d',
    ],
    # fp32: cast all (eligible) inputs to float32.
    'fp32': [
        'acos', 'asin', 'cosh', 'sinh', 'tan',
        'exp', 'log', 'rsqrt',
        'layer_norm', 'group_norm', 'nll_loss', 'mse_loss',
    ],
    # fp32_set_opt_dtype: fp32, flipping an unset output-dtype flag.
    'fp32_set_opt_dtype': ['softmax', 'log_softmax', 'sum', 'prod'],
    # promote: run in the widest input dtype.
    'promote': ['atan2'],
}

# CPU-only registrations, mirroring torch's KERNEL_CPU promote list.
AUTOCAST_POLICY_CPU_ONLY = {
    'promote': ['cat', 'stack'],
}


def autocast_policy_of(func_name, device_key=None):
    """Return the CastPolicy name for an op, or None.

    device_key selects 'CPU' or 'CUDA'; None matches either registration.
    """
    for policy, ops in AUTOCAST_POLICY.items():
        if func_name in ops:
            return policy
    if device_key == 'CPU':
        for policy, ops in AUTOCAST_POLICY_CPU_ONLY.items():
            if func_name in ops:
                return policy
    return None


def autocast_registered_ops():
    """Ops with an autocast kernel on any Autocast key."""
    ops = set()
    for policy_ops in AUTOCAST_POLICY.values():
        ops.update(policy_ops)
    for policy_ops in AUTOCAST_POLICY_CPU_ONLY.values():
        ops.update(policy_ops)
    return ops

# Type mapping for Python Interface (.pyi)
PYI_TYPE_MAP = {
    'int64_t[]': 'Sequence[int]',
    'str': 'str',
    'Tensor[]': 'Sequence[TensorBase]',
    'Tensor(a!)[]': 'Sequence[TensorBase]',
    'Scalar[]': 'Sequence[Scalar]',
    'DType': 'DType',
    'Device': 'Device',
    'Device?': 'Device | None',
    'double': 'float',
    'float': 'float',
    'bool': 'bool',
    'Scalar': 'Scalar',
    'Scalar?': 'Scalar | None',
    'Tensor': 'TensorBase',
    'Tensor(a!)': 'TensorBase',
    'int64_t': 'int',
    'int64_t?': 'int | None',
    'Tensor?': 'TensorBase | None',
}

def default_handler(type_str, default):
    if default == 'Float32': return 'DType::Float32'
    if default == 'CPU': return 'Device(DeviceType::CPU)'
    if default == 'Int64': return 'DType::Int64'
    if default == 'Undefined': return 'DType::Undefined'
    if default == 'None': return 'std::nullopt'
    if type_str == 'Scalar' and re.match(r'^-?\d+(\.\d+)?$', default):
        return default
    return default

def default_handler_pyi(type_str, default):
    if default == 'Float32' or default == 'DType::Float32': return 'DType.float32'
    if default == 'CPU' or default == 'Device(DeviceType::CPU)': return '...'
    if default == 'Int64' or default == 'DType::Int64': return 'DType.int64'
    if default == 'Undefined' or default == 'DType::Undefined': return 'DType.undefined'
    if default == 'None' or default == 'std::nullopt': return 'None'
    if default == 'true': return 'True'
    if default == 'false': return 'False'
    if type_str == 'Scalar' and re.match(r'^-?\d+(\.\d+)?$', default):
        return default
    # Tensor bias={} -> None (empty set literal is not a valid Tensor)
    if type_str == 'Tensor' and default == '{}':
        return 'None'
    # int64_t[] stride={1, 1, 1} -> tuple (a set literal would be unordered)
    if type_str.endswith('[]') and default.startswith('{') and default.endswith('}'):
        return '(' + default[1:-1] + ')'
    return default

def parse_func(func_str):
    arg_type_map = TYPE_MAP.copy()
    
    f = codegen_utils.parse_func(func_str, arg_type_map, default_handler)

    # Alias annotations are semantically important to Torch's schema but do
    # not change the generated C++ handle type.  Normalize every mutable
    # Tensor/TensorList alias to the canonical internal spelling so all
    # generators (device checks, version bumps, autograd edges, and bindings)
    # treat ``Tensor(b!)`` exactly like ``Tensor(a!)``.
    for arg in f['args']:
        canonical = re.sub(r'^Tensor\([A-Za-z_]\w*!\)$', 'Tensor(a!)', arg['type'])
        canonical = re.sub(r'^Tensor\([A-Za-z_]\w*!\)\[\]$', 'Tensor(a!)[]', canonical)
        if canonical != arg['type']:
            arg['type'] = canonical
            arg['cpp_type'] = TYPE_MAP[canonical]
    
    if f['schema_return_type'] == '()':
        f['return_type'] = 'void'
    elif f['schema_return_type'] == 'Tensor':
        f['return_type'] = 'Tensor'
    elif f['schema_return_type'] == 'Tensor(a!)':
        f['return_type'] = 'Tensor&'
    elif f['schema_return_type'] == 'Tensor[]':
        f['return_type'] = 'std::vector<Tensor>'
    elif f['schema_return_type'].startswith('(') and f['schema_return_type'].endswith(')'):
        # Handle tuple return type
         content = f['schema_return_type'][1:-1]
         if not content.strip():
             f['return_type'] = 'void'
         else:
             parts = [p.strip() for p in content.split(',')]
             cpp_types = []
             tuple_types = []
             return_names = []
             for p in parts:
                 tokens = p.split(' ')
                 type_part = tokens[0]
                 name_part = tokens[1] if len(tokens) > 1 else f"ret{len(tuple_types)}"

                 if type_part == 'Tensor':
                     cpp_types.append('Tensor')
                     tuple_types.append('Tensor')
                 else:
                     cpp_types.append(type_part)
                     tuple_types.append(type_part)
                 return_names.append(name_part)

             f['return_type'] = f"std::tuple<{', '.join(cpp_types)}>"
             f['is_tuple'] = True
             f['tuple_types'] = tuple_types
             f['return_names'] = return_names
        
    # Add stub_type
    for arg in f['args']:
        arg['stub_type'] = STUB_TYPE_MAP.get(arg['type'], arg['type'])
        
    return f

def sanitize_arg_name(name):
    if name == 'from': return 'from_'
    return name

def parse_dtypes_from_header(header_path):
    if not os.path.exists(header_path):
        return []
    
    with open(header_path, 'r') as f:
        content = f.read()
        
    match = re.search(r'enum class ScalarType\s*:\s*\w+\s*\{(.*?)\};', content, re.DOTALL)
    if not match:
        return []
        
    enum_body = match.group(1)
    dtypes = []
    current_val = 0
    
    for line in enum_body.split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        if line.endswith(','):
            line = line[:-1]
        if '=' in line:
            parts = line.split('=')
            name = parts[0].strip()
            val_str = parts[1].strip()
            try:
                current_val = int(val_str)
            except:
                pass
        else:
            name = line
            
        if name == 'NumOptions' or name == 'Undefined':
            pass
            
        py_name = name.lower()
        dtypes.append({'name': name, 'py_name': py_name, 'val': current_val})
        current_val += 1
        
    return dtypes

def generate_dtype_pyi(dtypes):
    lines = []
    lines.append("class DType(enum.Enum):")
    lines.append("    def __str__(self) -> str: ...")
    lines.append("    def __repr__(self) -> str: ...")
    lines.append("")
    for d in dtypes:
        lines.append(f"    {d['py_name']} = {d['val']}")
        lines.append("")
    for d in dtypes:
        lines.append(f"{d['py_name']}: DType = DType.{d['py_name']}")
        lines.append("")
    return "\n".join(lines)

def generate_pyi(funcs, template_path, dtype_header_path=None):
    with open(template_path, 'r') as f:
        template = f.read()

    methods_lines = []
    functions_lines = []
    
    for f in funcs:
        ret_type = PYI_TYPE_MAP.get(f['schema_return_type'], f['schema_return_type'])
        if f['schema_return_type'] == '()': ret_type = 'None'
        if f['schema_return_type'] == 'Tensor': ret_type = 'TensorBase'
        if f['schema_return_type'] == 'Tensor(a!)': ret_type = 'TensorBase'
        if f['schema_return_type'] == 'Tensor[]': ret_type = 'list[TensorBase]'
        if f['schema_return_type'] == 'int64_t[]': ret_type = 'Size'
        if f.get('is_tuple'):
            inner_types = []
            for t in f['tuple_types']:
                if t == 'Tensor':
                    inner_types.append('TensorBase')
                else:
                    inner_types.append(PYI_TYPE_MAP.get(t, t))
            ret_type = f"tuple[{', '.join(inner_types)}]"
        
        arg_strs = []
        start_idx = 0
        if f['variants'] == 'method':
             if f['args'] and f['args'][0]['name'] == 'self':
                 start_idx = 1

        for i in range(start_idx, len(f['args'])):
            arg = f['args'][i]
            py_type = PYI_TYPE_MAP.get(arg['type'], arg['type'])
            arg_name = sanitize_arg_name(arg['name'])
            s = f"{arg_name}: {py_type}"
            if arg['default']:
                default_val = default_handler_pyi(arg['type'], arg['default'])
                s += f" = {default_val}"
            arg_strs.append(s)
            
        sig = f"def {f['name']}({', '.join(arg_strs)}) -> {ret_type}: ..."
        
        if f['variants'] == 'method':
            methods_lines.append(f"    {sig}")
        else:
            functions_lines.append(f"{sig}")
            
    template = template.replace("${generated_methods}", "\n".join(methods_lines))
    template = template.replace("${generated_functions}", "\n".join(functions_lines))
    
    if dtype_header_path:
        dtypes = parse_dtypes_from_header(dtype_header_path)
        dtype_str = generate_dtype_pyi(dtypes)
        template = template.replace("${generated_dtypes}", dtype_str)
    
    return template

# Tensor-returning Tensor methods mapped to their tpx::ops free functions.
_TENSOR_METHODS = {
    'neg': 'neg', 't': 't', 'mm': 'mm', 'matmul': 'matmul', 'transpose': 'transpose',
    'squeeze': 'squeeze', 'unsqueeze': 'unsqueeze', 'permute': 'permute', 'view': 'view',
    'reshape': 'reshape', 'expand': 'expand', 'sum': 'sum', 'mean': 'mean', 'pow': 'pow',
    'sqrt': 'sqrt', 'sin': 'sin', 'cos': 'cos', 'exp': 'exp', 'log': 'log', 'tanh': 'tanh',
    'sigmoid': 'sigmoid', 'relu': 'relu', 'softmax': 'softmax', 'log_softmax': 'log_softmax',
    'abs': 'abs', 'square': 'square', 'sign': 'sign', 'mul': 'mul', 'add': 'add', 'sub': 'sub', 'div': 'div',
    'atan2': 'atan2', 'clamp': 'clamp', 'lerp': 'lerp', 'clone': 'clone', 'detach': 'detach',
    'contiguous': 'contiguous', 'select': 'select', 'slice': 'slice', 't_': 't_',
}

# Methods that return non-tensor metadata: keep the p10 method call as-is.
_META_METHODS = {
    'shape', 'numel', 'size', 'item', 'dim', 'ndimension', 'storage_offset',
    'is_contiguous', 'dtype', 'device', 'sizes', 'strides', 'requires_grad',
    'grad_fn', 'stride', 'to', 'type', 'eq', 'ne', 'lt', 'le', 'gt', 'ge',
}

# Helper functions invoked in derivatives.yaml that are numeric kernels (not
# autograd building blocks); their arguments are still translated.
_FUNC_CALLS = {
    'clamp_backward', 'threshold_backward', 'nll_loss_backward', 'mse_loss_backward',
    'max_pool2d_backward', 'adaptive_avg_pool2d_backward', 'adaptive_max_pool2d_backward',
    'batch_norm_backward', 'layer_norm_backward', 'group_norm_backward',
    'instance_norm_backward', 'constant_pad_nd_backward', 'conv2d_grad_input',
    'conv2d_grad_weight', 'conv2d_grad_bias', 'conv1d_grad_input', 'conv1d_grad_weight',
    'conv1d_grad_bias', 'conv3d_grad_input', 'conv3d_grad_weight', 'conv3d_grad_bias',
    'conv_transpose2d_grad_input', 'conv_transpose2d_grad_weight', 'conv_transpose2d_grad_bias',
    'conv_transpose3d_grad_input', 'conv_transpose3d_grad_weight', 'conv_transpose3d_grad_bias',
    'embedding_dense_backward', 'permute_backward', 'squeeze_backward',
}


def _split_plus_minus(s):
    # Split on + / - at paren-depth 0, but keep unary +/- (after an operator,
    # '(', ',', or at the start) attached to its operand.
    depth = 0
    parts = []
    cur = []
    for ch in s:
        if ch in '([{':
            depth += 1
            cur.append(ch)
        elif ch in ')]}':
            depth -= 1
            cur.append(ch)
        elif depth == 0 and ch in '+-':
            t = ''.join(cur).rstrip()
            prev_ch = t[-1] if t else None
            if prev_ch is None or prev_ch in '*/(+-':
                cur.append(ch)
                continue
            parts.append((t, ch))
            cur = []
        else:
            cur.append(ch)
    parts.append((''.join(cur), None))
    return parts


def _split_top_level(s, ops):
    depth = 0
    parts = []
    cur = []
    for ch in s:
        if ch == '(' or ch == '[' or ch == '{':
            depth += 1
            cur.append(ch)
        elif ch == ')' or ch == ']' or ch == '}':
            depth -= 1
            cur.append(ch)
        elif depth == 0 and ch in ops:
            parts.append((''.join(cur), ch))
            cur = []
        else:
            cur.append(ch)
    parts.append((''.join(cur), None))
    return parts


def _split_args(s):
    parts = _split_top_level(s, ',')
    return [p[0].strip() for p in parts[:-1]] + [parts[-1][0].strip()] if parts else []


def _is_number(s):
    s = s.strip()
    return re.fullmatch(r'-?\d+(\.\d+)?([eE][+-]?\d+)?', s) is not None


def _is_tensor_expr(s, tensor_params):
    s = s.strip()
    if not s:
        return False
    if s.startswith('(') and s.endswith(')') and _is_balanced(s):
        return _is_tensor_expr(s[1:-1], tensor_params)
    if s in tensor_params or s in ('grad', 'grad_output', 'result'):
        return True
    # std::get<N>(...)
    if re.match(r'^std::get<\d+>\(', s):
        return True
    # tensorplay::tpx::ops::* or *_backward(...) or plain helper(...)
    m = re.match(r'^([A-Za-z_]\w*)(::)?(?:[A-Za-z_]\w*)?(?:::ops::)?([A-Za-z_]\w*)\(', s)
    if m and (s.startswith('tensorplay::tpx::ops::') or re.match(r'^[A-Za-z_]\w*\(', s)):
        # A bare function call whose name is not a constructor/scalar
        name = re.match(r'^([A-Za-z_]\w*)\(', s).group(1)
        if name not in ('Scalar',):
            return True
    # Method calls returning a tensor (e.g. self.pow(...)) count as tensor exprs
    m = re.match(r'^[A-Za-z_]\w*\.([A-Za-z_]\w*)\(', s)
    if m and m.group(1) in _TENSOR_METHODS:
        return True
    return False


def _looks_tensor_like(s, tensor_params):
    s = s.strip()
    if not s:
        return False
    if _is_tensor_expr(s, tensor_params):
        return True
    if re.search(r'[.*/]', s):
        return True
    if s.startswith('(') and _is_balanced(s):
        return True
    if re.match(r'^[A-Za-z_]\w*(\(|\.)', s):
        return True
    return False


def _translate_expr(s, tensor_params):
    s = s.strip()
    if not s:
        return s

    # Parenthesized expression
    if s.startswith('(') and s.endswith(')'):
        depth = 0
        for i, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    break
        else:
            return _translate_expr(s[1:-1], tensor_params)

    # Unary minus
    if s.startswith('-'):
        inner = _translate_expr(s[1:], tensor_params)
        if _looks_tensor_like(s[1:], tensor_params):
            return f'neg({inner})'
        return f'-{inner}'

    # Binary + / - (lowest precedence)
    parts = _split_plus_minus(s)
    if len(parts) > 2 or (len(parts) == 2 and parts[0][0].strip()):
        expr = _translate_expr(parts[0][0], tensor_params)
        for i in range(1, len(parts)):
            op = parts[i - 1][1]
            part = parts[i][0]
            r = _translate_expr(part, tensor_params)
            if _is_tensor_expr(parts[0][0], tensor_params):
                fn = 'add' if op == '+' else 'sub'
                expr = f'{fn}({expr}, {r})'
            else:
                # Scalar op Tensor: only '-' needs reordering; '+' and scalars
                # stay as-is (Scalar arithmetic). If right is tensor and left
                # scalar, mirror the expression.
                if _looks_tensor_like(part, tensor_params) and op == '-':
                    expr = f'neg(sub({r}, {expr}))'
                else:
                    expr = f'{expr} {op} {r}'
        return expr

    # Binary * / (next precedence)
    parts = _split_top_level(s, '*/')
    if len(parts) > 1:
        expr = _translate_expr(parts[0][0], tensor_params)
        for i in range(1, len(parts)):
            op = parts[i - 1][1]
            part = parts[i][0]
            r = _translate_expr(part, tensor_params)
            if _is_tensor_expr(parts[0][0], tensor_params):
                fn = 'mul' if op == '*' else 'div'
                expr = f'{fn}({expr}, {r})'
            elif _is_tensor_expr(part, tensor_params) and op == '*':
                # scalar * tensor -> mul(tensor, scalar)
                expr = f'mul({r}, {expr})'
            else:
                expr = f'{expr} {op} {r}'
        return expr

    # std::get<N>(...) keep wrapper, translate inner args
    m = re.match(r'^(std::get<\d+>)\((.*)\)$', s, re.S)
    if m:
        inner = ', '.join(_translate_expr(a, tensor_params) for a in _split_args(m.group(2)))
        return f'{m.group(1)}({inner})'

    # (expr).method(...) chain
    m = re.match(r'^\(.*\)\.([A-Za-z_]\w*)\(', s)
    if m:
        depth = 0
        end = None
        for j, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is not None:
            obj = _translate_expr(s[1:end], tensor_params)
            return _translate_tail(s[end + 1:], obj, tensor_params)

    # Method / function call chain
    if s.endswith(')') and not s.startswith('('):
        m = re.match(r'^([A-Za-z_]\w*)\.([A-Za-z_]\w*)\(', s)
        if m:
            parts = _find_call_parts(s)
            if parts:
                head, args, rest = parts
                obj, method = head.rsplit('.', 1)
                inner = ', '.join(_translate_expr(a, tensor_params) for a in _split_args(args)) if args.strip() else ''
                if method in _TENSOR_METHODS:
                    translated = f'{_TENSOR_METHODS[method]}({obj}, {inner})' if inner else f'{_TENSOR_METHODS[method]}({obj})'
                else:
                    translated = f'{obj}.{method}({inner})' if inner else f'{obj}.{method}()'
                return _translate_tail(rest, translated, tensor_params)

        m = re.match(r'^([A-Za-z_]\w*)\(', s)
        if m:
            parts = _find_call_parts(s)
            if parts:
                head, args, rest = parts
                inner = ', '.join(_translate_expr(a, tensor_params) for a in _split_args(args)) if args.strip() else ''
                if head == 'Scalar':
                    translated = f'Scalar({inner})' if inner else 'Scalar()'
                else:
                    translated = f'{head}({inner})'
                return _translate_tail(rest, translated, tensor_params)

    return s


def _find_call_parts(s):
    # For 'obj.method(args)rest' or 'name(args)rest', return (head, args, rest)
    # using balanced-paren matching so nested calls in args work.
    i = s.find('(')
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == '(':
            depth += 1
        elif s[j] == ')':
            depth -= 1
            if depth == 0:
                return s[:i], s[i + 1:j], s[j + 1:]
    return None


def _translate_tail(rest, base, tensor_params):
    # Handle remaining chained method calls like .exp() or .sum({dim}, true)
    rest = rest.strip()
    if not rest:
        return base
    m = re.match(r'^\.([A-Za-z_]\w*)\(', rest)
    if not m:
        return base + rest
    parts = _find_call_parts(rest)
    if not parts:
        return base + rest
    _, args, more = parts
    inner = ', '.join(_translate_expr(a, tensor_params) for a in _split_args(args)) if args.strip() else ''
    method = m.group(1)
    if method in _TENSOR_METHODS:
        translated = f'{_TENSOR_METHODS[method]}({base}, {inner})' if inner else f'{_TENSOR_METHODS[method]}({base})'
    else:
        translated = f'{base}.{method}({inner})' if inner else f'{base}.{method}()'
    return _translate_tail(more, translated, tensor_params)


def _is_balanced(s):
    depth = 0
    for ch in s:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def generate_autograd_nodes(derivatives, native_funcs):
    lines = []
    lines.append("// Generated by tools/codegen/gen.py")
    lines.append("#pragma once")
    lines.append("#include \"Node.h\"")
    lines.append("#include \"Autograd.h\"")
    lines.append("#include \"ManualNodes.h\"")
    lines.append("#include \"tensorplay/ops/TPXOpsGenerated.h\"")
    lines.append("#include <algorithm>")
    lines.append("#include <utility>")
    lines.append("#include \"Scalar.h\"")
    lines.append("#include <vector>")
    lines.append("#include <cstdint>")
    lines.append("#include <cstdio>")
    lines.append("")
    lines.append("namespace tensorplay {")
    lines.append("namespace tpx {")
    lines.append("using namespace ops;")
    lines.append("")

    native_map = {f['func_name']: f for f in native_funcs}

    emitted_node_names = set()
    for d in derivatives:
        func_name = parse_func(d['name'])['func_name']
        if func_name not in native_map:
            continue
        # SDPA has a tuple-valued fused backward implementation and its
        # hand-written node evaluates that tuple once.  Do not emit a second
        # generated class with the same name.
        if func_name in {'scaled_dot_product_attention', 'mean'}:
            continue
            
        native_f = native_map[func_name]
        node_name = autograd_node_name(func_name)
        # ``add_.Tensor``/``add.Tensor`` (and the other in-place arithmetic
        # overloads) intentionally share a backward node.  The generated
        # wrapper can reuse the first definition.
        if node_name in emitted_node_names:
            continue
        emitted_node_names.add(node_name)
        
        formulas = {}
        for arg in native_f['args']:
            if arg['name'] in d:
                formulas[arg['name']] = d[arg['name']]
        
        if native_f.get('is_tuple'):
            for name in native_f['return_names']:
                if name in d:
                    formulas[name] = d[name]
        
        # Backward formulas may refer not only to forward inputs, but also to
        # forward outputs.  ``result`` is the output name for scalar-returning
        # operators and tuple-returning operators expose their named outputs
        # (for example ``total_weight`` in nll_loss).  Keep those symbols when
        # collecting node state so the generated node constructor and formula
        # agree with the generated call site.
        arg_names = {a['name'] for a in native_f['args']}
        output_names = {'result'}
        if native_f.get('is_tuple'):
            output_names.update(native_f.get('return_names', []))
        symbol_names = arg_names | output_names
        used_vars = set()
        for formula in formulas.values():
            words = re.findall(r'\b[a-zA-Z_]\w*\b', formula)
            for w in words:
                if w in symbol_names:
                    used_vars.add(w)
        
        members = []
        for arg in native_f['args']:
            if arg['name'] in used_vars:
                cpp_type = "Tensor" if arg['type'] in ['Tensor', 'Tensor(a!)'] else arg['type']
                if cpp_type == 'Tensor': cpp_type = 'Tensor'
                if arg['type'] in ['Scalar', 'double', 'float', 'int64_t']:
                    cpp_type = arg['cpp_type']
                if arg['type'] == 'int64_t[]':
                    cpp_type = 'std::vector<int64_t>'
                if arg['type'] == 'int64_t?':
                    cpp_type = 'std::optional<int64_t>'
                if arg['type'] == 'Scalar?':
                    cpp_type = 'std::optional<Scalar>'
                if arg['type'] == 'Tensor?':
                    cpp_type = 'std::optional<Tensor>'
                members.append({'name': arg['name'], 'type': cpp_type})
        
        if native_f.get('is_tuple'):
            for i, name in enumerate(native_f['return_names']):
                if name in used_vars:
                    t_type = native_f['tuple_types'][i]
                    cpp_type = "Tensor" if t_type == 'Tensor' else t_type
                    members.append({'name': name, 'type': cpp_type})
        else:
            if 'result' in used_vars:
                cpp_type = "Tensor"
                if native_f['return_type'] == 'std::vector<Tensor>':
                    cpp_type = 'std::vector<Tensor>'
                members.append({'name': 'result', 'type': cpp_type})
        
        lines.append(f"struct {node_name} : public Node {{")
        for m in members:
            lines.append(f"    {m['type']} {m['name']}_;")
        lines.append("")
        
        ctor_args = []
        ctor_inits = []
        for m in members:
            ctor_args.append(f"{m['type']} {m['name']}")
            ctor_inits.append(f"{m['name']}_({m['name']})")
            
        lines.append(f"    explicit {node_name}({', '.join(ctor_args)})")
        if ctor_inits:
            lines.append(f"        : {', '.join(ctor_inits)} {{}}")
        else:
            lines.append(f"        {{}}")
        lines.append("")
        lines.append("    variable_list apply(variable_list&& inputs) override {")
        # Optional Tensor arguments still occupy an autograd edge/gradient
        # slot.  Omitting them makes the returned variable_list shorter than
        # collect_next_edges(...), so parameters after the first optional
        # input silently receive no gradient (notably BatchNorm weight/bias).
        tensor_arg_types = ['Tensor', 'Tensor(a!)', 'Tensor?']
        n_tensor_args = sum(1 for arg in native_f['args'] if arg['type'] in tensor_arg_types)
        undefined = ", ".join(["Tensor()"] * n_tensor_args)
        lines.append(f"        if (inputs.empty() || !inputs[0].defined()) return {{{undefined}}};")
        lines.append("        const Tensor& grad = inputs[0];")
        
        lines.append("")
        lines.append("        variable_list grads;")

        # BatchNorm's three derivative formulas all invoke the same backward
        # kernel.  Compute its tuple once; otherwise adding the optional
        # weight/bias gradient slots would accidentally make every BatchNorm
        # backward pass three times more expensive.
        if func_name == 'batch_norm':
            lines.append(
                "        auto batch_norm_backward_result = batch_norm_backward("
                "grad, input_, weight_, running_mean_, running_var_, training_, eps_);"
            )
        
        for arg in native_f['args']:
            if arg['type'] in tensor_arg_types:
                if arg['name'] in formulas:
                    if func_name == 'batch_norm' and arg['name'] in {'input', 'weight', 'bias'}:
                        result_index = {'input': 0, 'weight': 1, 'bias': 2}[arg['name']]
                        lines.append(
                            f"        grads.push_back(std::get<{result_index}>"
                            "(batch_norm_backward_result));"
                        )
                        continue
                    formula = formulas[arg['name']]
                    tensor_params = {a['name'] for a in native_f['args'] if 'Tensor' in a['type']}
                    formula = _translate_expr(formula, tensor_params)
                    for m in members:
                        # Avoid replacing method calls (e.g. shape() vs shape)
                        formula = re.sub(r'\b' + m['name'] + r'\b(?!\()', m['name'] + '_', formula)
                    lines.append(f"        grads.push_back({formula});")
                else:
                    lines.append(f"        grads.push_back(Tensor());")

        lines.append("        return grads;")
        lines.append("    }")
        lines.append("};")
        lines.append("")

    lines.append("} // namespace tpx")
    lines.append("} // namespace tensorplay")
    return "\n".join(lines)

def generate_header(funcs):
    lines = []
    lines.append("// Generated by tools/codegen/gen.py")
    lines.append("#pragma once")
    lines.append("#include <tuple>")
    lines.append("")
    
    for f in funcs:
        sig = f['return_type'] + " " + f['name'] + "("
        arg_strs = []
        is_const_method = False
        if f['variants'] == 'method':
             self_arg = next((arg for arg in f['args'] if arg['name'] == 'self'), None)
             if self_arg and '!' not in self_arg['type']:
                 is_const_method = True

        for arg in f['args']:
            if f['variants'] == 'method' and arg['name'] == 'self':
                continue
            if arg['name'] == 'requires_grad': continue
            s = f"{arg['cpp_type']} {arg['name']}"
            if arg['default']:
                s += f" = {arg['default']}"
            arg_strs.append(s)
        sig += ", ".join(arg_strs) + ")"
        
        if is_const_method:
            sig += " const"
        
        if f['variants'] == 'function':
            lines.append(f"static {sig};")
        else:
            lines.append(f"{sig};")
        lines.append("")
        
    return "\n".join(lines)

def has_autograd_logic(f):
    """True when the generated tpx wrapper for this op builds autograd graph
    nodes (i.e. an autograd kernel should be registered and consulted)."""
    return bool(f.get('autograd')) or f.get('func_name') == 'relu_'


def autograd_node_name(func_name):
    """Return the Torch-style backward node name for an overload.

    In-place arithmetic has the same backward node as its functional
    overload (``add_.Tensor`` -> ``AddTensorBackward``).  Treating the
    mutable marker as an overload spelling also prevents duplicate node
    definitions in the generated header.
    """
    canonical = func_name.replace('_.', '.')
    clean_name = "".join(x.title() for x in canonical.replace('.', '_').split('_'))
    return clean_name + "Backward"

def generate_cpp(funcs):
    lines = []
    lines.append("// Generated by tools/codegen/gen.py")
    lines.append("#include \"Tensor.h\"")
    lines.append("#include \"tensorplay/ops/TensorRedispatchGenerated.h\"")
    lines.append("#include \"Dispatcher.h\"")
    lines.append("#include \"Exception.h\"")
    lines.append("#include \"DispatchKey.h\"")
    lines.append("#include \"GradMode.h\"")
    lines.append("#include \"DType.h\"")
    lines.append("#include \"Scalar.h\"")
    lines.append("#include \"SizesAndStrides.h\"")
    lines.append("#include \"Device.h\"")
    lines.append("#include \"TypePromotion.h\"")
    lines.append("#include \"autocast_mode.h\"")
    lines.append("#ifdef USE_CUDA")
    lines.append("#include \"CUDARuntime.h\"")
    lines.append("#endif")
    lines.append("#include <tuple>")
    lines.append("#include <utility>")
    lines.append("")
    lines.append("namespace tensorplay {")
    lines.append("")
    
    for f in funcs:
        if f.get('skip_implementation'): continue
        
        sig = f['return_type'] + " Tensor::" + f['name'] + "("
        arg_strs = []
        is_const_method = False
        if f['variants'] == 'method':
             self_arg = next((arg for arg in f['args'] if arg['name'] == 'self'), None)
             if self_arg and '!' not in self_arg['type']:
                 is_const_method = True

        for arg in f['args']:
            if f['variants'] == 'method' and arg['name'] == 'self':
                continue
            if arg['name'] == 'requires_grad': continue
            s = f"{arg['cpp_type']} {arg['name']}"
            arg_strs.append(s)
        sig += ", ".join(arg_strs) + ")"
        
        if is_const_method:
            sig += " const"

        # Pure Dispatch - No Autograd
        dispatch_key_source = "Device(DeviceType::CPU)"
        device_arg = next((arg for arg in f['args'] if arg['name'] == 'device'), None)

        target_device_expr = "Device(DeviceType::CPU)"

        if device_arg:
            if device_arg['type'] == 'Device?':
                if f['name'].endswith('_like'):
                    self_arg = next((arg for arg in f['args'] if arg['name'] == 'self'), None)
                    if self_arg:
                        if f['variants'] == 'method':
                             # In method variant, self is 'this', accessible via device()
                             val = f"{device_arg['name']}.has_value() ? *{device_arg['name']} : device()"
                        else:
                             val = f"{device_arg['name']}.has_value() ? *{device_arg['name']} : {self_arg['name']}.device()"
                        dispatch_key_source = val
                        target_device_expr = val
                    else:
                        val = f"{device_arg['name']}.has_value() ? *{device_arg['name']} : Device(DeviceType::CPU)"
                        dispatch_key_source = val
                        target_device_expr = val
                else:
                    val = f"{device_arg['name']}.has_value() ? *{device_arg['name']} : Device(DeviceType::CPU)"
                    dispatch_key_source = val
                    target_device_expr = val
            else:
                dispatch_key_source = "device"
                target_device_expr = "device"
        elif f['variants'] == 'method':
            dispatch_key_source = "device()"
            target_device_expr = "device()"
        else:
            # For functions without explicit device arg, use the first tensor argument's device
            first_tensor_arg = next((arg for arg in f['args'] if arg['type'] in ['Tensor', 'Tensor(a!)']), None)
            if first_tensor_arg:
                dispatch_key_source = f"{first_tensor_arg['name']}.device()"
                target_device_expr = f"{first_tensor_arg['name']}.device()"
            else:
                # Tensor-list operators (cat/stack/etc.) do not have a
                # scalar Tensor argument.  Their dispatch device must follow
                # the first defined element instead of silently defaulting to
                # CPU, which would reject every CUDA list.
                first_tensor_list = next(
                    (arg for arg in f['args'] if arg['type'] in ('Tensor[]', 'Tensor(a!)[]')),
                    None,
                )
                if first_tensor_list:
                    val = (f"({first_tensor_list['name']}.empty() ? "
                           "Device(DeviceType::CPU) : "
                           f"{first_tensor_list['name']}[0].device())")
                    dispatch_key_source = val
                    target_device_expr = val
            
        # ------------------------------------------------------------------
        # 1) Backend redispatch entry point (emitted before the method).
        #
        # Mirrors PyTorch's at::redispatch:: namespace: a free function that
        # calls the registered BACKEND kernel directly, skipping any
        # autograd kernel. Autograd wrappers (registered under AutogradCPU /
        # AutogradCUDA) funnel through this so re-entering the Tensor method
        # cannot recurse.
        # ------------------------------------------------------------------
        rd_args = []
        rd_call_args = []
        template_args = [f['return_type']]
        for arg in f['args']:
            if arg['name'] == 'requires_grad': continue
            template_args.append(arg['stub_type'])
            rd_args.append(f"{arg['stub_type']} {arg['name']}")
            rd_call_args.append(arg['name'])
        template_str = ", ".join(template_args)
        rd_call_str = ", ".join(rd_call_args)

        # Inside the free function there is no `this`; method variants take
        # `self` as an explicit first parameter instead.
        if f['variants'] == 'method':
            rd_device_source = dispatch_key_source.replace('device()', 'self.device()')
        else:
            rd_device_source = dispatch_key_source

        # Version bumps for mutated tensors live in redispatch so that both
        # the plain path and the autograd path (wrapper -> redispatch)
        # record mutations.
        mutable_exprs = [
            arg['name'] for arg in f['args']
            if arg['type'] in ('Tensor(a!)', 'Tensor(a!)[]')
        ]

        # Function and method variants can have the same backend signature
        # (for example mm(Tensor, Tensor)).  Keep their public overloads, but
        # give the internal redispatch helpers variant-specific names so the
        # generated translation unit never defines the same helper twice.
        redispatch_name = f"redispatch_{f['name']}_{f['variants']}"
        lines.append("namespace detail {")
        lines.append(f"TENSORPLAY_API {f['return_type']} {redispatch_name}({', '.join(rd_args)}) {{")
        lines.append("#ifdef USE_CUDA")
        lines.append(f"    cuda::OptionalCUDAGuard device_guard({rd_device_source});")
        lines.append("#endif")
        lines.append(
            f'    static const OperatorHandle op_handle = '
            f'Dispatcher::singleton().findHandle("{f["func_name"]}");'
        )
        lines.append(
            f"    DispatchKey dispatch_key = computeDispatchKey({rd_device_source});")

        if not mutable_exprs:
            if f['return_type'] == 'void':
                lines.append(f"    DispatchStub<{template_str}>::call(op_handle, dispatch_key, {rd_call_str});")
                lines.append("    return;")
            else:
                lines.append(f"    return DispatchStub<{template_str}>::call(op_handle, dispatch_key, {rd_call_str});")
        else:
            if f['return_type'] == 'void':
                lines.append(f"    DispatchStub<{template_str}>::call(op_handle, dispatch_key, {rd_call_str});")
            else:
                lines.append(f"    auto&& __tp_result = DispatchStub<{template_str}>::call(op_handle, dispatch_key, {rd_call_str});")
            for expr in mutable_exprs:
                arg_type = next(arg['type'] for arg in f['args'] if arg['name'] == expr)
                if arg_type == 'Tensor(a!)[]':
                    lines.append(f"    for (const auto& __tp_tensor : {expr}) {{")
                    lines.append("        if (__tp_tensor.defined()) __tp_tensor.unsafeGetTensorImpl()->bump_version();")
                    lines.append("    }")
                else:
                    lines.append(f"    {expr}.unsafeGetTensorImpl()->bump_version();")
            if f['return_type'] == 'void':
                lines.append("    return;")
            else:
                lines.append("    return std::forward<decltype(__tp_result)>(__tp_result);")
        lines.append("}")
        lines.append("} // namespace detail")
        lines.append("")

        # ------------------------------------------------------------------
        # 2) Tensor method: device checks -> autograd key -> redispatch.
        # ------------------------------------------------------------------
        # 2) Tensor method: device checks -> autograd key -> redispatch.
        # ------------------------------------------------------------------
        lines.append(sig + " {")

        if f['name'] == 'copy_':
            lines.append('    if (!impl_ || !src.impl_) TP_THROW(RuntimeError, "Tensor not defined");')
            lines.append('    if (this->shape() != src.shape()) {')
            lines.append('        TP_THROW(RuntimeError, "copy_(): shapes mismatch (broadcasting not yet supported)");')
            lines.append('    }')

        # Device Check (skip for copy_ which allows cross-device)
        if f['name'] != 'copy_' and f.get('device_check') != 'NoCheck':
            # Special handling for factory functions like empty_like, zeros_like, etc.
            # where 'self' is used for metadata but 'device' argument dictates output device.
            # We should not enforce self.device() == target_device if target_device is explicitly provided (or default).
            is_factory_like = f['name'].endswith('_like')

            for arg in f['args']:
                if f['variants'] == 'method' and arg['name'] == 'self': continue

                # Skip check for 'self' in *_like functions
                if is_factory_like and arg['name'] == 'self': continue

                if arg['type'] in ['Tensor', 'Tensor(a!)', 'Tensor?', 'Tensor[]', 'Tensor(a!)[]']:
                    if arg['type'] == 'Tensor?':
                        lines.append(f"    if ({arg['name']}.has_value() && {arg['name']}->defined() && {arg['name']}->device() != {target_device_expr}) {{")
                        lines.append(f'        TP_THROW(DeviceMismatchError, "Expected all tensors to be on the same device, but found one ({arg["name"]}) on " + {arg["name"]}->device().toString() + " and another ({target_device_expr}) on " + {target_device_expr}.toString());')
                        lines.append("    }")
                    elif arg['type'] in ['Tensor[]', 'Tensor(a!)[]']:
                        lines.append(f"    for (const auto& t : {arg['name']}) {{")
                        lines.append(f"        if (t.defined() && t.device() != {target_device_expr}) {{")
                        lines.append(f'            TP_THROW(DeviceMismatchError, "Expected all tensors to be on the same device, but found one (in {arg["name"]}) on " + t.device().toString() + " and another ({target_device_expr}) on " + {target_device_expr}.toString());')
                        lines.append("        }")
                        lines.append("    }")
                    else:
                        lines.append(f"    if ({arg['name']}.defined() && {arg['name']}.device() != {target_device_expr}) {{")
                        lines.append(f'        TP_THROW(DeviceMismatchError, "Expected all tensors to be on the same device, but found one ({arg["name"]}) on " + {arg["name"]}.device().toString() + " and another ({target_device_expr}) on " + {target_device_expr}.toString());')
                        lines.append("    }")

        method_call_args = []
        for arg in f['args']:
            if arg['name'] == 'requires_grad': continue
            if f['variants'] == 'method' and arg['name'] == 'self':
                method_call_args.append("*this")
            else:
                method_call_args.append(arg['name'])
        method_call_str = ", ".join(method_call_args)

        # Route through the dispatcher's autocast kernel when one is
        # registered for this op and autocast is enabled for the op's device.
        # The Autocast key outranks the Autograd key (mirroring PyTorch, where
        # "autocasting precedes VariableType"), so casts are autograd-exposed
        # and inputs are saved for backward in the post-cast type.
        if f['func_name'] in autocast_registered_ops():
            lines.append("    {")
            lines.append(
                '        static const OperatorHandle __ac_handle = '
                f'Dispatcher::singleton().findHandle("{f["func_name"]}");')
            lines.append(
                f"        DispatchKey __ac_key = toAutocastKey(computeDispatchKey({dispatch_key_source}));")
            lines.append("        if (__ac_handle && __ac_handle.getKernel(__ac_key) && autocast::is_enabled(__ac_key)) {")
            if f['return_type'] == 'void':
                lines.append(
                    f"            DispatchStub<{template_str}>::call(__ac_handle, __ac_key, {method_call_str});")
                lines.append("            return;")
            else:
                lines.append(
                    f"            return DispatchStub<{template_str}>::call(__ac_handle, __ac_key, {method_call_str});")
            lines.append("        }")
            lines.append("    }")

        # Route through the dispatcher's autograd kernel when one is
        # registered for this op and recording is enabled. The kernel itself
        # decides whether any input actually requires grad (mirrors PyTorch,
        # where the Autograd key is always consulted first under GradMode).
        if has_autograd_logic(f):
            lines.append("    if (GradMode::is_enabled()) {")
            lines.append(
                f'        static const OperatorHandle ag_handle = '
                f'Dispatcher::singleton().findHandle("{f["func_name"]}");')
            lines.append(f"        DispatchKey ag_key = toAutogradKey(computeDispatchKey({dispatch_key_source}));")
            lines.append("        if (ag_handle && ag_handle.getKernel(ag_key)) {")
            if f['return_type'] == 'void':
                lines.append(f"            DispatchStub<{template_str}>::call(ag_handle, ag_key, {method_call_str});")
                lines.append("            return;")
            else:
                lines.append(f"            return DispatchStub<{template_str}>::call(ag_handle, ag_key, {method_call_str});")
            lines.append("        }")
            lines.append("    }")

        if f['return_type'] == 'void':
            lines.append(f"    detail::{redispatch_name}({method_call_str});")
        else:
            lines.append(f"    return detail::{redispatch_name}({method_call_str});")
        lines.append("}")
        lines.append("")

    lines.append("} // namespace tensorplay")
    return "\n".join(lines)

def _autocast_arg_expr(policy, arg):
    """Per-argument expression inside an autocast kernel, mirroring the
    cached_cast / set_opt_dtype application of torch's WrapFunction_."""
    if policy in ('lower_precision_fp', 'fp32', 'promote'):
        return (f"::tensorplay::autocast::cached_cast(__to_type, {arg['name']}, "
                "__device_type)")
    if policy == 'fp32_set_opt_dtype':
        if arg['type'] == 'DType':
            return (f"::tensorplay::autocast::set_opt_dtype(DType::Float32, "
                    f"{arg['name']})")
        return arg['name']
    return arg['name']


def generate_autocast_registration(funcs):
    """Emit autocast kernels and their AutocastCPU/AutocastCUDA registrations.

    Mirrors the KERNEL_CPU/KERNEL_CUDA registrations of
    aten/src/ATen/autocast_mode.cpp: each wrapper applies its CastPolicy to
    the arguments (through ExcludeAutocastGuard so nested dispatch cannot
    recurse) and redispatches to the tpx::ops wrapper, which routes through
    autograd with post-cast dtypes.
    """
    lines = []
    lines.append("// Generated by tools/codegen/gen.py")
    lines.append("#include \"Dispatcher.h\"")
    lines.append("#include \"DispatchKey.h\"")
    lines.append("#include \"Device.h\"")
    lines.append("#include \"DType.h\"")
    lines.append("#include \"autocast_mode.h\"")
    lines.append("#include \"autocast_cast.h\"")
    lines.append("#include \"tensorplay/ops/TPXOpsGenerated.h\"")
    lines.append("")
    lines.append("namespace tensorplay {")
    lines.append("namespace {")
    lines.append("")

    seen = set()
    kernels = []  # (op_name, kernel_symbol, device_key)
    for f in funcs:
        if f.get('skip_implementation'):
            continue
        name = f['func_name']
        if name in seen:
            continue

        for device_key in ('CPU', 'CUDA'):
            policy = autocast_policy_of(name, device_key)
            if policy is None:
                continue
            if name not in seen:
                seen.add(name)

            kernel = f"autocast_kernel_{name}_{device_key.lower()}"
            kernels.append((name, kernel, device_key))

            arg_strs = []
            for arg in f['args']:
                if arg['name'] == 'requires_grad':
                    continue
                arg_strs.append(f"{arg['cpp_type']} {arg['name']}")
            sig = f"{f['return_type']} {kernel}({', '.join(arg_strs)})"

            lines.append(sig + " {")
            lines.append(f"    const DeviceType __device_type = DeviceType::{device_key};")
            lines.append(
                "    ::tensorplay::autocast::ExcludeAutocastGuard no_autocast(__device_type);")

            call_args = [a for a in f['args'] if a['name'] != 'requires_grad']
            call_str = ", ".join(_autocast_arg_expr(policy, a) for a in call_args)

            if policy in ('lower_precision_fp', 'fp32'):
                if policy == 'lower_precision_fp':
                    lines.append(
                        "    const DType __to_type = "
                        "::tensorplay::autocast::get_lower_precision_fp_from_device_type(__device_type);")
                else:
                    lines.append("    const DType __to_type = DType::Float32;")
                if f['return_type'] == 'void':
                    lines.append(f"    ::tensorplay::tpx::ops::{name}({call_str});")
                else:
                    lines.append(f"    return ::tensorplay::tpx::ops::{name}({call_str});")
            elif policy == 'fp32_set_opt_dtype':
                all_args_str = ", ".join(a['name'] for a in call_args)
                plain_str = ", ".join(a['name'] for a in call_args)
                lines.append(
                    "    if (::tensorplay::autocast::firstarg_is_eligible(__device_type, "
                    f"{all_args_str})) {{")
                if f['return_type'] == 'void':
                    lines.append(f"        ::tensorplay::tpx::ops::{name}({call_str});")
                    lines.append("        return;")
                else:
                    lines.append(f"        return ::tensorplay::tpx::ops::{name}({call_str});")
                lines.append("    }")
                if f['return_type'] == 'void':
                    lines.append(f"    ::tensorplay::tpx::ops::{name}({plain_str});")
                else:
                    lines.append(f"    return ::tensorplay::tpx::ops::{name}({plain_str});")
            elif policy == 'promote':
                all_args_str = ", ".join(a['name'] for a in call_args)
                lines.append(
                    "    const DType __to_type = ::tensorplay::autocast::promote_type(")
                lines.append(
                    "        ::tensorplay::autocast::get_lower_precision_fp_from_device_type(__device_type),")
                lines.append(f"        __device_type, {all_args_str});")
                if f['return_type'] == 'void':
                    lines.append(f"    ::tensorplay::tpx::ops::{name}({call_str});")
                else:
                    lines.append(f"    return ::tensorplay::tpx::ops::{name}({call_str});")
            lines.append("}")
            lines.append("")

    lines.append("} // anonymous namespace")
    lines.append("")

    for device_key, lib_name in (('CPU', 'AutocastKernelsCPU'), ('CUDA', 'AutocastKernelsCUDA')):
        lines.append(f"TENSORPLAY_LIBRARY_IMPL({device_key}, {lib_name}) {{")
        for name, kernel, key in kernels:
            if key != device_key:
                continue
            lines.append(f'    m.impl("{name}", &{kernel});')
        lines.append("}")
        lines.append("")

    lines.append("} // namespace tensorplay")
    lines.append("")
    return "\n".join(lines)


def generate_redispatch_header(funcs):
    """Generate declarations shared by p10's methods and tpx's wrappers."""
    lines = [
        "// Generated by tools/codegen/gen.py",
        "#pragma once",
        "#include \"Tensor.h\"",
        "#include \"Macros.h\"",
        "",
        "namespace tensorplay {",
        "namespace detail {",
        "",
    ]

    for f in funcs:
        if f.get('skip_implementation'):
            continue
        rd_args = []
        for arg in f['args']:
            if arg['name'] == 'requires_grad':
                continue
            rd_args.append(f"{arg['stub_type']} {arg['name']}")
        redispatch_name = f"redispatch_{f['name']}_{f['variants']}"
        lines.append(
            f"TENSORPLAY_API {f['return_type']} {redispatch_name}({', '.join(rd_args)});"
        )
        lines.append("")

    lines.extend([
        "} // namespace detail",
        "} // namespace tensorplay",
    ])
    return "\n".join(lines)

def generate_tpx_ops_h(funcs):
    lines = []
    lines.append("// Generated by tools/codegen/gen.py")
    lines.append("#pragma once")
    lines.append("#include \"Autograd.h\"")
    lines.append("#include <tuple>")
    lines.append("")
    lines.append("namespace tensorplay {")
    lines.append("namespace tpx {")
    lines.append("namespace ops {")
    lines.append("")
    
    seen_sigs = set()
    
    for f in funcs:
        # TPX Ops are always free functions
        # For methods, self becomes first argument
        sig = "TENSORPLAY_API " + f['return_type'] + " " + f['name'] + "("
        arg_strs = []
        arg_types = []
        
        # Reconstruct args including self
        for arg in f['args']:
            s = f"{arg['cpp_type']} {arg['name']}"
            if arg['default']:
                s += f" = {arg['default']}"
            arg_strs.append(s)
            arg_types.append(arg['cpp_type'])
            
        sig += ", ".join(arg_strs) + ")"
        
        dedup_key = f['name'] + ":" + ",".join(arg_types)
        if dedup_key in seen_sigs:
            continue
        seen_sigs.add(dedup_key)

        lines.append(f"{sig};")
        lines.append("")
        
    lines.append("} // namespace ops")
    lines.append("} // namespace tpx")
    lines.append("} // namespace tensorplay")
    return "\n".join(lines)

def generate_tpx_ops_cpp(funcs):
    lines = []
    lines.append("// Generated by tools/codegen/gen.py")
    lines.append("#include \"Autograd.h\"")
    lines.append("#include \"tensorplay/ops/AutogradNodesGenerated.h\"")
    lines.append("#include \"Node.h\"")
    lines.append("#include \"tensorplay/ops/TPXOpsGenerated.h\"")
    lines.append("#include \"tensorplay/ops/TensorRedispatchGenerated.h\"")
    lines.append("#include \"Dispatcher.h\"")
    lines.append("#include \"DispatchKey.h\"")
    lines.append("#include \"autocast_mode.h\"")
    lines.append("#include <algorithm>")
    lines.append("")
    lines.append("namespace tensorplay {")
    lines.append("namespace tpx {")
    lines.append("namespace ops {")
    lines.append("")
    
    seen_sigs = set()
    
    for f in funcs:
        sig = f['return_type'] + " " + f['name'] + "("
        arg_strs = []
        arg_types = []
        for arg in f['args']:
            s = f"{arg['cpp_type']} {arg['name']}"
            arg_strs.append(s)
            arg_types.append(arg['cpp_type'])
        sig += ", ".join(arg_strs) + ")"
        
        dedup_key = f['name'] + ":" + ",".join(arg_types)
        if dedup_key in seen_sigs:
            continue
        seen_sigs.add(dedup_key)
        lines.append(sig + " {")

        # Route through the dispatcher's autocast kernel before any autograd
        # handling (Autocast outranks Autograd, mirroring PyTorch).  The
        # autocast kernel casts eligible inputs and re-enters this wrapper
        # with autocast excluded, so recording sees post-cast dtypes.
        if f['func_name'] in autocast_registered_ops():
            ac_template_args = [f['return_type']] + arg_types
            ac_template_str = ", ".join(ac_template_args)
            ac_call_str = ", ".join(a['name'] for a in f['args'])
            ac_dev_source = None
            for a in f['args']:
                if a['type'] in ('Tensor', 'Tensor(a!)'):
                    ac_dev_source = f"{a['name']}.device()"
                    break
            if ac_dev_source is None:
                for a in f['args']:
                    if a['type'] in ('Tensor[]', 'Tensor(a!)[]'):
                        ac_dev_source = (f"({a['name']}.empty() ? "
                                         "Device(DeviceType::CPU) : "
                                         f"{a['name']}[0].device())")
                        break
            if ac_dev_source is not None:
                lines.append("    {")
                lines.append(
                    '        static const OperatorHandle __ac_handle = '
                    'Dispatcher::singleton().findHandle('
                    f'"{f["func_name"]}");')
                lines.append(
                    "        DispatchKey __ac_key = toAutocastKey(computeDispatchKey("
                    f"{ac_dev_source}));")
                lines.append(
                    "        if (__ac_handle && __ac_handle.getKernel(__ac_key) && ::tensorplay::autocast::is_enabled(__ac_key)) {")
                if f['return_type'] == 'void':
                    lines.append(
                        f"            DispatchStub<{ac_template_str}>::call(__ac_handle, __ac_key, {ac_call_str});")
                    lines.append("            return;")
                else:
                    lines.append(
                        f"            return DispatchStub<{ac_template_str}>::call(__ac_handle, __ac_key, {ac_call_str});")
                lines.append("        }")
                lines.append("    }")


        # relu_ is an aliasing operation, so its derivative uses the same
        # post-ReLU result mask as relu, but the generated generic wrapper
        # historically skipped all autograd handling for Tensor& returns.
        # Keep this small exception here instead of introducing a duplicate
        # ReluBackward node in derivatives.yaml.
        is_inplace_relu = f.get('func_name') == 'relu_'
        has_autograd = bool(f.get('autograd')) or is_inplace_relu
        
        # Check requires_grad
        bool_requires_grad_decl = ""
        if has_autograd:
            bool_requires_grad_decl = '    bool requires_grad = false;'
            lines.append(bool_requires_grad_decl)
            tensor_args_check = []
            for arg in f['args']:
                if arg['type'] in ['Tensor', 'Tensor(a!)']:
                    tensor_args_check.append(f"{arg['name']}.requires_grad()")
                elif arg['type'] == 'Tensor[]':
                    tensor_args_check.append(
                        f"std::any_of({arg['name']}.begin(), {arg['name']}.end(), "
                        "[](const Tensor& tensor) { return tensor.requires_grad(); })")
                elif arg['type'] == 'Tensor?':
                    tensor_args_check.append(f"({arg['name']}.has_value() && {arg['name']}->requires_grad())")
             
            if tensor_args_check:
                cond = " || ".join(tensor_args_check)
                lines.append(f"    if (GradMode::is_enabled() && ({cond})) requires_grad = true;")

        if has_autograd:
            # Torch allows differentiable in-place arithmetic only on
            # non-leaf tensors.  The normal optimizer path runs with
            # GradMode disabled and therefore never enters this branch.
            for arg in f['args']:
                if arg['type'] != 'Tensor(a!)':
                    continue
                arg_name = arg['name']
                lines.append(
                    "    if (requires_grad && " + arg_name + ".requires_grad() && tensorplay::tpx::impl::is_leaf(" + arg_name + ")) {"
                )
                lines.append(
                    '        TP_THROW(RuntimeError, "a leaf Variable that requires grad is being used in an in-place operation");'
                )
                lines.append("    }")

        # Call underlying p10 Tensor method or function
        call_args = []
        for arg in f['args']:
            if f['variants'] == 'method' and arg['name'] == 'self':
                continue
            call_args.append(arg['name'])
        
        call_args_str = ", ".join(call_args)

        # Core call: hit the backend kernel through the generated redispatch
        # entry point. Going through detail::redispatch_* (instead of the
        # Tensor method) is what breaks the recursion: this wrapper IS the
        # autograd kernel for the op, so re-entering the method would find
        # the autograd key again and loop forever.
        #
        # Ops with skip_implementation have no generated method/dispatcher
        # entry (e.g. view() is hand-written on p10::Tensor); keep calling
        # the Tensor method directly for those.
        if f.get('skip_implementation'):
            if f['variants'] == 'method':
                self_arg = f['args'][0]['name']
                core_call_args = []
                for arg in f['args']:
                    if arg['name'] in ('self', 'requires_grad'): continue
                    core_call_args.append(arg['name'])
                call_line = f"{self_arg}.{f['name']}({', '.join(core_call_args)})"
            else:
                core_call_args = [arg['name'] for arg in f['args'] if arg['name'] != 'requires_grad']
                call_line = f"Tensor::{f['name']}({', '.join(core_call_args)})"
        else:
            core_call_args = []
            for arg in f['args']:
                if arg['name'] == 'requires_grad': continue
                core_call_args.append(arg['name'])
            core_call_str = ", ".join(core_call_args)
            redispatch_name = f"redispatch_{f['name']}_{f['variants']}"
            call_line = f"::tensorplay::detail::{redispatch_name}({core_call_str})"

        # Execute call
        if f.get('is_tuple'):
            lines.append(f"    auto core_result = {call_line};")
        elif f['return_type'] == 'Tensor':
            lines.append(f"    Tensor result = {call_line};")
        elif f['return_type'] == 'std::vector<Tensor>':
            lines.append(f"    auto core_result = {call_line};")
        elif f['return_type'] == 'Tensor&':
             lines.append(f"    {call_line};")
        elif f['return_type'] == 'void':
             lines.append(f"    {call_line};")
        else:
             # void or scalar
             pass

        # Autograd Node Creation (After call)
        if has_autograd:
            lines.append("    std::shared_ptr<Node> grad_fn;")
            lines.append("    if (requires_grad) {")
             
            node_cls = 'ReluBackward' if is_inplace_relu else f.get('autograd_node_name', f['autograd'][0])
            autograd_args_spec = f.get('autograd_args_spec', [])
            
            node_arg_list = []
            if is_inplace_relu:
                # The in-place result receives this node's grad_fn below. Do
                # not save that same Tensor in the node or it forms a
                # Tensor -> Node -> Tensor reference cycle across batches.
                node_arg_list.append(f"{f['args'][0]['name']}.detach()")
            elif autograd_args_spec:
                for arg_spec in autograd_args_spec:
                    if arg_spec['source'] == 'input':
                        node_arg_list.append(arg_spec['name'])
                    elif arg_spec['source'] == 'output':
                        if 'index' in arg_spec:
                            idx = arg_spec['index']
                            output_expr = f"std::get<{idx}>(core_result)"
                        else:
                            output_expr = "result"
                        # ReluBackward saves the output mask. Save an
                        # autograd-detached view so attaching grad_fn to the
                        # returned result cannot create a reference cycle.
                        if node_cls == 'ReluBackward':
                            output_expr += ".detach()"
                        node_arg_list.append(output_expr)
            
            node_args = ", ".join(node_arg_list)
            lines.append(f"        grad_fn = std::make_shared<{node_cls}>({node_args});")
            
            edge_args = []
            for arg in f['args']:
                if arg['type'] in ['Tensor', 'Tensor(a!)']:
                    edge_args.append(arg['name'])
                elif arg['type'] in ['Tensor[]', 'Tensor(a!)[]']:
                    edge_args.append(arg['name'])
                elif arg['type'] == 'Tensor?':
                    edge_args.append(arg['name'])

            if any(arg['type'] in ['Tensor[]', 'Tensor(a!)[]'] for arg in f['args']):
                lines.append("        std::vector<Edge> autograd_edges;")
                for arg in f['args']:
                    if arg['type'] in ['Tensor[]', 'Tensor(a!)[]']:
                        lines.append(f"        for (const auto& tensor : {arg['name']}) {{")
                        lines.append("            collect_next_edges_helper(autograd_edges, tensor);")
                        lines.append("        }")
                    elif arg['type'] in ['Tensor', 'Tensor(a!)']:
                        lines.append(f"        collect_next_edges_helper(autograd_edges, {arg['name']});")
                    elif arg['type'] == 'Tensor?':
                        lines.append(f"        collect_next_edges_helper(autograd_edges, {arg['name']});")
                lines.append("        grad_fn->add_next_edge_list(std::move(autograd_edges));")
            else:
                edge_args_str = ", ".join(edge_args)
                lines.append(f"        grad_fn->add_next_edge_list(collect_next_edges({edge_args_str}));")
            lines.append("    }")

        # Wrap result
        if f.get('is_tuple'):
            lines.append(f"    std::tuple<{', '.join(f['tuple_types'])}> result;")
            for i, t_type in enumerate(f['tuple_types']):
                if t_type == 'Tensor':
                    lines.append(f"    std::get<{i}>(result) = std::get<{i}>(core_result);")
                    if f.get('autograd'):
                        lines.append(f"    if (requires_grad && std::get<{i}>(result).defined()) {{")
                        lines.append(f"        tensorplay::tpx::impl::set_grad_fn(std::get<{i}>(result), grad_fn, {i});")
                        lines.append(f"    }}")
                else:
                    lines.append(f"    std::get<{i}>(result) = std::get<{i}>(core_result);")
            lines.append("    return result;")
            
        elif f['return_type'] == 'Tensor':
            # Check for explicit requires_grad argument (for factories)
            has_requires_grad_arg = False
            for arg in f['args']:
                if arg['name'] == 'requires_grad':
                    has_requires_grad_arg = True
                    break
            
            if has_requires_grad_arg:
                lines.append("    tensorplay::tpx::impl::set_requires_grad(result, requires_grad);")

            # Set history
            if has_autograd:
                lines.append("    if (requires_grad) tensorplay::tpx::impl::set_requires_grad(result, true);")
                lines.append("    if (requires_grad && result.defined()) {")
                lines.append("        tensorplay::tpx::impl::set_grad_fn(result, grad_fn);")
                lines.append("    }")
            
            lines.append("    return result;")
            
        elif f['return_type'] == 'std::vector<Tensor>':
             # core_result is vector<Tensor>
             lines.append("    return core_result;")
             
        elif f['return_type'] == 'Tensor&':
             # Alias returns belong to the mutable argument, which is not
             # necessarily the first schema argument for out= overloads.
             first_arg = next(
                 (arg['name'] for arg in f['args'] if arg['type'] == 'Tensor(a!)'),
                 f['args'][0]['name'],
             )
             if has_autograd:
                 lines.append(f"    if (requires_grad) tensorplay::tpx::impl::set_requires_grad({first_arg}, true);")
                 lines.append(f"    if (requires_grad && {first_arg}.defined()) {{")
                 lines.append(f"        tensorplay::tpx::impl::set_grad_fn({first_arg}, grad_fn);")
                 lines.append("    }")
             lines.append(f"    return {first_arg};")
             
        else:
            if f['return_type'] == 'void':
                lines.append("    return;")
            else:
                lines.append(f"    return {call_line};")
            
        lines.append("}")
        lines.append("")
        
    lines.append("} // namespace ops")
    lines.append("} // namespace tpx")
    lines.append("} // namespace tensorplay")
    return "\n".join(lines)

def get_tpx_ops_signature(f):
    # Free function signature
    ret_type = f['return_type']
    arg_types = [arg['cpp_type'] for arg in f['args']]
    args_str = ", ".join(arg_types)
    return f"{ret_type} (*)({args_str})"

def transform_binding_default(val, cpp_type):
    if val == 'CPU': return 'Device(DeviceType::CPU)'
    if val == 'Undefined': return 'DType::Undefined'
    if val == 'None': return 'py::none()'
    if val.startswith('{'):
        type_name = cpp_type.replace('const ', '').replace('&', '').strip()
        return f'{type_name}{val}'
    if val == '[]':
        # Python literal list default -> C++ brace init of the vector type
        type_name = cpp_type.replace('const ', '').replace('&', '').strip()
        return f'{type_name}{{}}'
    return val

def generate_bindings(funcs):
    lines = []
    lines.append("// Generated by tools/codegen/gen.py")
    lines.append("#pragma once")
    lines.append("#include <pybind11/pybind11.h>")
    lines.append("#include <pybind11/stl.h>")
    lines.append("#include \"Autograd.h\"")
    lines.append("#include \"tensorplay/ops/TPXOpsGenerated.h\"")
    lines.append("")
    lines.append("namespace py = pybind11;")
    lines.append("")
    lines.append("namespace tensorplay {")
    lines.append("namespace python {")
    lines.append("")
    lines.append("using Tensor = tensorplay::Tensor;")
    lines.append("")
    lines.append("inline void bind_generated_tensor_methods(py::class_<Tensor>& m) {")
    
    methods_by_name = {}
    for f in funcs:
        if f['variants'] == 'method':
            name = f['name']
            if name not in methods_by_name:
                methods_by_name[name] = []
            methods_by_name[name].append(f)

    for name, method_list in methods_by_name.items():
        is_overloaded = True 
        
        for f in method_list:
            args_list = []
            for arg in f['args']:
                if arg['name'] == 'self': continue
                arg_name = sanitize_arg_name(arg['name'])
                s = f'py::arg("{arg_name}")'
                if arg['default']:
                    default_val = transform_binding_default(arg['default'], arg['cpp_type'])
                    s += f" = {default_val}"
                args_list.append(s)
            
            args_str = ", ".join(args_list)
            if args_str:
                args_str = ", " + args_str
                
            cast_str = ""
            if is_overloaded:
                sig = get_tpx_ops_signature(f)
                cast_str = f"static_cast<{sig}>"
            
            # Bind to tpx::ops::name
            ptr_str = f"&tensorplay::tpx::ops::{f['name']}"
            if is_overloaded:
                ptr_str = f"{cast_str}({ptr_str})"
                
            lines.append(f'    m.def("{f["name"]}", {ptr_str}{args_str});')
            
    lines.append("}")
    lines.append("")
    lines.append("inline void bind_generated_op_functions(py::module_& m) {")
    
    functions_by_name = {}
    for f in funcs:
        if f['variants'] == 'function':
            name = f['name']
            if name not in functions_by_name:
                functions_by_name[name] = []
            functions_by_name[name].append(f)

    for name, function_list in functions_by_name.items():
        is_overloaded = True
        
        for f in function_list:
            args_list = []
            for arg in f['args']:
                arg_name = sanitize_arg_name(arg['name'])
                s = f'py::arg("{arg_name}")'
                if arg['default']:
                    default_val = transform_binding_default(arg['default'], arg['cpp_type'])
                    s += f" = {default_val}"
                args_list.append(s)
            
            args_str = ", ".join(args_list)
            if args_str:
                args_str = ", " + args_str
                
            cast_str = ""
            if is_overloaded:
                sig = get_tpx_ops_signature(f)
                cast_str = f"static_cast<{sig}>"
            
            # Bind to tpx::ops::name
            ptr_str = f"&tensorplay::tpx::ops::{f['name']}"
            if is_overloaded:
                ptr_str = f"{cast_str}({ptr_str})"
                
            lines.append(f'    m.def("{f["name"]}", {ptr_str}{args_str});')

    lines.append("}")
    lines.append("} // namespace python")
    lines.append("} // namespace tensorplay")
    return "\n".join(lines)

def generate_functional_py(funcs):
    lines = []
    lines.append("# Generated by tools/codegen/gen.py")
    lines.append("# Do not edit directly")
    lines.append("")
    lines.append("import tensorplay")
    lines.append("import tensorplay._C as _C")
    lines.append("from tensorplay._C import DType")
    lines.append("from tensorplay.compiler.graph import capture_call as _capture_call")
    lines.append("")
    lines.append("def _ensure_device(device):")
    lines.append("    if device is None or device is Ellipsis:")
    lines.append("        return tensorplay.device(\"cpu\")")
    lines.append("    if isinstance(device, str):")
    lines.append("        return tensorplay.device(device)")
    lines.append("    return device")
    lines.append("")

    seen_funcs = set()
    has_matmul_out = any(f.get('func_name') == 'matmul.out' for f in funcs)
    
    for f in funcs:
        variants = [v.strip() for v in f['variants'].split(',')]
        name = f['name']
        if name in seen_funcs:
            continue

        # ``where`` is an overload family in Torch: tensor/tensor, scalar/
        # tensor, tensor/scalar and scalar/scalar.  A name-only de-duplication
        # would silently expose only the first schema (the old generator did
        # exactly that), making Python numbers fail before the dispatcher can
        # select the native overload.  Keep one public wrapper and let the
        # generated extension perform overload resolution after normalizing
        # Python numbers to TensorPlay::Scalar.
        if name == 'where' and 'function' in variants:
            seen_funcs.add(name)
            lines.append("def where(condition, input, other):")
            lines.append("    _captured = _capture_call(where, (condition, input, other), {})")
            lines.append("    if _captured is not None:")
            lines.append("        return _captured")
            lines.append("    _input_is_tensor = isinstance(input, tensorplay.Tensor)")
            lines.append("    _other_is_tensor = isinstance(other, tensorplay.Tensor)")
            lines.append("    if not _input_is_tensor:")
            lines.append("        input = tensorplay.Scalar(input)")
            lines.append("    if not _other_is_tensor:")
            lines.append("        other = tensorplay.Scalar(other)")
            lines.append("    return _C.where(condition=condition, self=input, other=other)")
            lines.append("")
            continue

        # Foreach operators are overload families.  Torch selects the
        # overload from the second/third argument (Scalar, Tensor[],
        # Scalar[], ...), so a fixed two-argument Python signature would
        # hide most of the dispatcher surface.  Keep this wrapper variadic
        # and pass the call straight to the generated native overload set;
        # the implementation remains in the dispatcher/backend, not in a
        # Python fallback loop.
        if name.startswith('_foreach_') and 'function' in variants:
            seen_funcs.add(name)
            lines.append(f"def {name}(input, *args, **kwargs):")
            lines.append(
                f"    _captured = _capture_call({name}, (input, *args), kwargs)"
            )
            lines.append("    if _captured is not None:")
            lines.append("        return _captured")
            lines.append(f"    return _C.{name}(input, *args, **kwargs)")
            lines.append("")
            continue
            
        if 'function' in variants:
            if name in ['randn', 'rand', 'zeros', 'ones', 'empty']:
                seen_funcs.add(name)
                supports_pin_memory = name in ['zeros', 'ones', 'empty']
                pin_parameter = ", pin_memory=False" if supports_pin_memory else ""
                lines.append(f"def {name}(*size, dtype=DType.float32, device=None{pin_parameter}, requires_grad=False):")
                lines.append("    if len(size) == 1 and (isinstance(size[0], (list, tuple)) or hasattr(size[0], '__iter__')):")
                lines.append("        _size = size[0]")
                lines.append("    else:")
                lines.append("        _size = size")
                pin_argument = ", pin_memory=pin_memory" if supports_pin_memory else ""
                capture_kwargs = (
                    "{'dtype': dtype, 'device': device"
                    + (", 'pin_memory': pin_memory" if supports_pin_memory else "")
                    + ", 'requires_grad': requires_grad}"
                )
                lines.append(
                    f"    _captured = _capture_call({name}, tuple(size), {capture_kwargs})"
                )
                lines.append("    if _captured is not None:")
                lines.append("        return _captured")
                lines.append(f"    return _C.{name}(size=list(_size), dtype=dtype, device=_ensure_device(device){pin_argument}, requires_grad=requires_grad)")
                lines.append("")
                continue

            if name == 'arange':
                seen_funcs.add(name)
                lines.append("def arange(*args, dtype=DType.undefined, device=None, requires_grad=False):")
                lines.append("    _captured = _capture_call(arange, tuple(args), {'dtype': dtype, 'device': device, 'requires_grad': requires_grad})")
                lines.append("    if _captured is not None:")
                lines.append("        return _captured")
                lines.append("    if len(args) == 1:")
                lines.append("        return _C.arange(end=args[0], dtype=dtype, device=_ensure_device(device), requires_grad=requires_grad)")
                lines.append("    elif len(args) == 2:")
                lines.append("        return _C.arange(start=args[0], end=args[1], dtype=dtype, device=_ensure_device(device), requires_grad=requires_grad)")
                lines.append("    elif len(args) == 3:")
                lines.append("        return _C.arange(start=args[0], end=args[1], step=args[2], dtype=dtype, device=_ensure_device(device), requires_grad=requires_grad)")
                lines.append("    else:")
                lines.append("        raise TypeError(f'arange expected 1-3 positional arguments, got {len(args)}')")
                lines.append("")
                continue

            seen_funcs.add(name)

            # Torch exposes matmul's out variant as a keyword-only optional
            # argument on the functional API, while the native schema keeps
            # it as a separate aliasing overload. Keep graph capture on the
            # differentiable two-input path; out= is eager and follows
            # Torch's no-autograd contract.
            if name == 'matmul' and has_matmul_out:
                lines.append("def matmul(input, other, *, out=None):")
                lines.append("    if out is not None:")
                lines.append("        return _C.matmul(self=input, other=other, out=out)")
                lines.append("    _captured = _capture_call(matmul, (input, other), {})")
                lines.append("    if _captured is not None:")
                lines.append("        return _captured")
                lines.append("    return _C.matmul(self=input, other=other)")
                lines.append("")
                continue

            arg_strs = []
            call_args = []
            
            args = f['args']
            
            for arg in args:
                arg_name = sanitize_arg_name(arg['name'])
                param_name = arg_name
                # Map 'self' to 'input' for consistency with PyTorch functional API
                if param_name == 'self':
                    param_name = 'input'
                
                s = f"{param_name}"
                if arg['default']:
                     default_val = default_handler_pyi(arg['type'], arg['default'])
                     s += f"={default_val}"
                arg_strs.append(s)
                
                # Pass arguments to _C function using keyword arguments
                # The C++ binding uses the original argument name (sanitized)
                if arg_name == 'device':
                    if name.endswith('_like'):
                        call_args.append(f"{arg_name}={param_name}")
                    else:
                        call_args.append(f"{arg_name}=_ensure_device({param_name})")
                else:
                    call_args.append(f"{arg_name}={param_name}")
            
            sig_args = ", ".join(arg_strs)
            call_args_str = ", ".join(call_args)
            
            lines.append(f"def {name}({sig_args}):")

            # Functional wrappers are part of the compiler's operator
            # surface.  During capture, route symbolic calls into the
            # canonical graph instead of passing Proxy objects through
            # pybind11 into the eager extension.  At runtime the same wrapper
            # is called with real values, so this is transparent to eager
            # execution and keeps generated code as the source of truth.
            capture_params = []
            for arg in args:
                param_name = sanitize_arg_name(arg['name'])
                if param_name == 'self':
                    param_name = 'input'
                capture_params.append(param_name)
            capture_tuple = ", ".join(capture_params)
            if len(capture_params) == 1:
                capture_tuple += ","
            lines.append(
                f"    _captured = _capture_call({name}, ({capture_tuple}), {{}})"
            )
            lines.append("    if _captured is not None:")
            lines.append("        return _captured")

            # Check for Scalar arguments and convert them
            for arg in args:
                # if name == 'full':
                #     print(f"DEBUG: Processing full, arg={arg['name']}, type={arg['type']}")
                if arg['type'] == 'Scalar':
                    param_name = sanitize_arg_name(arg['name'])
                    if param_name == 'self': param_name = 'input'
                    lines.append(f"    if not isinstance({param_name}, (tensorplay.Scalar, tensorplay.Tensor)):")
                    lines.append(f"        {param_name} = tensorplay.Scalar({param_name})")
                elif arg['type'] == 'Scalar?':
                    param_name = sanitize_arg_name(arg['name'])
                    if param_name == 'self': param_name = 'input'
                    lines.append(f"    if {param_name} is not None and not isinstance({param_name}, (tensorplay.Scalar, tensorplay.Tensor)):")
                    lines.append(f"        {param_name} = tensorplay.Scalar({param_name})")
                elif arg['type'] == 'Tensor':
                    param_name = sanitize_arg_name(arg['name'])
                    if param_name == 'self': param_name = 'input'
                    if arg['default'] == '{}':
                        lines.append(f"    if {param_name} is None:")
                        lines.append(f"        {param_name} = tensorplay.Tensor()")

            lines.append(f"    return _C.{name}({call_args_str})")
            lines.append("")
            
        elif 'method' in variants:
            # Fallback for method-only variants
            seen_funcs.add(name)
            
            arg_strs = []
            call_args = []
            
            args = f['args']
            
            if not args or args[0]['name'] != 'self':
                continue
                
            arg_strs.append("input")
            
            for i in range(1, len(args)):
                arg = args[i]
                arg_name = sanitize_arg_name(arg['name'])
                s = f"{arg_name}"
                if arg['default']:
                     default_val = default_handler_pyi(arg['type'], arg['default'])
                     s += f"={default_val}"
                arg_strs.append(s)
                call_args.append(f"{arg_name}={arg_name}")
            
            sig_args = ", ".join(arg_strs)
            call_args_str = ", ".join(call_args)
            
            lines.append(f"def {name}({sig_args}):")
            if not name.endswith("_"):
                # Method-only native definitions (for example add/mul/div)
                # still have a public functional wrapper.  Capture that
                # wrapper before dispatching to the Tensor method so the
                # compiler sees one canonical call_function node instead of a
                # pybind call with symbolic arguments.
                capture_params = ["input"]
                capture_params.extend(
                    sanitize_arg_name(arg['name'])
                    for arg in args[1:]
                )
                capture_tuple = ", ".join(capture_params)
                if len(capture_params) == 1:
                    capture_tuple += ","
                lines.append(
                    f"    _captured = _capture_call({name}, ({capture_tuple}), {{}})"
                )
                lines.append("    if _captured is not None:")
                lines.append("        return _captured")
            lines.append(f"    return input.{name}({call_args_str})")
            lines.append("")
            
    return "\n".join(lines)

def generate_autograd_registration(funcs):
    """Emits the translation unit that registers every generated tpx wrapper
    as the AutogradCPU/AutogradCUDA kernel of its operator, mirroring PyTorch's
    RegisterAutogradCUDA.cpp / RegisterAutogradCPU.cpp."""
    lines = []
    lines.append("// Generated by tools/codegen/gen.py")
    lines.append("#include \"Dispatcher.h\"")
    lines.append("#include \"DispatchKey.h\"")
    lines.append("#include \"tensorplay/ops/TPXOpsGenerated.h\"")
    lines.append("")
    lines.append("namespace tensorplay {")
    lines.append("namespace {")
    lines.append("")
    lines.append("struct RegisterTPXAutogradKernels {")
    lines.append("    RegisterTPXAutogradKernels() {")
    lines.append("        auto& D = Dispatcher::singleton();")

    seen_sigs = set()
    for f in funcs:
        if f.get('skip_implementation'): continue
        if not has_autograd_logic(f): continue

        # Mirror generate_tpx_ops_cpp dedup: only register overloads whose
        # wrapper was actually emitted.
        arg_types = [arg['cpp_type'] for arg in f['args']]
        dedup_key = f['name'] + ":" + ",".join(arg_types)
        if dedup_key in seen_sigs:
            continue
        seen_sigs.add(dedup_key)

        # Registration must match the C++ wrapper's exact ABI.  In
        # particular Tensor? is represented as const std::optional<Tensor>&
        # in the generated tpx function, while stub_type intentionally uses
        # the by-value form for dispatcher templates.
        ret = f['return_type']
        param_types = [arg['cpp_type'] for arg in f['args']]
        fn_type = f"{ret} (*)({', '.join(param_types)})"
        cast = f"static_cast<{fn_type}>(&::tensorplay::tpx::ops::{f['name']})"
        op_name = f["func_name"]
        for key in ("AutogradCPU", "AutogradCUDA"):
            lines.append(f'        D.registerKernel("{op_name}", DispatchKey::{key}, (KernelFunction){cast});')

    lines.append("    }")
    lines.append("};")
    lines.append("")
    lines.append("static RegisterTPXAutogradKernels g_register_tpx_autograd_kernels;")
    lines.append("")
    lines.append("} // namespace")
    lines.append("} // namespace tensorplay")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', required=True, help='Path to native_functions.yaml')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--pyi_template', help='Path to _C.pyi.in')
    parser.add_argument('--pyi_out', help='Path to output _C.pyi')
    parser.add_argument('--derivatives', help='Path to derivatives.yaml')
    parser.add_argument('--pkg_out', help='Output directory for package')
    args = parser.parse_args()
    
    derivatives_map = {}
    if args.derivatives and os.path.exists(args.derivatives):
        with open(args.derivatives, 'r') as f:
            d_data = yaml.load(f, Loader=YamlLoader)
            if d_data:
                for item in d_data:
                    try:
                        d_f = parse_func(item['name'])
                        derivatives_map[d_f['func_name']] = item
                    except:
                        pass
    
    with open(args.yaml, 'r') as f:
        data = yaml.load(f, Loader=YamlLoader)
        
    if data is None:
        data = []
        
    funcs = []
    for item in data:
        base_f = parse_func(item['func'])
        base_f['autograd'] = item.get('autograd')
        
        if base_f['func_name'] in derivatives_map:
            func_name = base_f['func_name']
            node_name = autograd_node_name(func_name)
            base_f['autograd_node_name'] = node_name
            base_f['autograd'] = [node_name]
            
            d = derivatives_map[func_name]
            formulas = {}
            for arg in base_f['args']:
                if arg['name'] in d:
                    formulas[arg['name']] = d[arg['name']]
            
            used_vars = set()
            for formula in formulas.values():
                words = re.findall(r'\b[a-zA-Z_]\w*\b', formula)
                for w in words:
                    if w in ['grad', 'grad_output', 'neg', 'pow', 'sin', 'cos', 'exp', 'log', 'tanh', 't', 'mm', 'div', 'mul', 'add', 'sub']: continue
                    used_vars.add(w)
            
            autograd_args_spec = []
            for arg in base_f['args']:
                if arg['name'] in used_vars:
                    autograd_args_spec.append({'name': arg['name'], 'source': 'input'})
            
            if base_f.get('is_tuple'):
                 for i, name in enumerate(base_f['return_names']):
                     if name in used_vars:
                         autograd_args_spec.append({'name': name, 'source': 'output', 'index': i})
            else:
                 if 'result' in used_vars:
                     autograd_args_spec.append({'name': 'result', 'source': 'output'})
            
            base_f['autograd_args_spec'] = autograd_args_spec
            base_f['autograd_args'] = [x['name'] for x in autograd_args_spec]

        # Tensor-list view ops have no scalar derivative formula to express in
        # derivatives.yaml, but their backward is a cheap slice/select and is
        # required by RoPE/concatenation-heavy decoder graphs.
        if base_f['func_name'] in ('cat', 'stack'):
            base_f['autograd_node_name'] = 'CatBackward' if base_f['func_name'] == 'cat' else 'StackBackward'
            base_f['autograd'] = [base_f['autograd_node_name']]
            base_f['autograd_args_spec'] = [
                {'name': 'tensors', 'source': 'input'},
                {'name': 'dim', 'source': 'input'},
            ]
                 
        base_f['dispatch'] = item.get('dispatch')
        base_f['device_check'] = item.get('device_check')
        base_f['skip_implementation'] = item.get('skip_implementation', False)
        
        variants_str = item.get('variants', 'function')
        variants = [v.strip() for v in variants_str.split(',')]
        
        for v in variants:
            f = copy.deepcopy(base_f)
            f['variants'] = v
            funcs.append(f)
        
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    header_path = os.path.join(args.out_dir, "TensorGenerated.h")
    cpp_path = os.path.join(args.out_dir, "TensorGenerated.cpp")
    bindings_path = os.path.join(args.out_dir, "TensorBindingsGenerated.h")
    autograd_nodes_path = os.path.join(args.out_dir, "AutogradNodesGenerated.h")
    tpx_ops_h_path = os.path.join(args.out_dir, "TPXOpsGenerated.h")
    tpx_ops_cpp_path = os.path.join(args.out_dir, "TPXOpsGenerated.cpp")
    tpx_autograd_reg_path = os.path.join(args.out_dir, "TPXAutogradRegistration.cpp")
    autocast_reg_path = os.path.join(args.out_dir, "AutocastGenerated.cpp")
    redispatch_h_path = os.path.join(args.out_dir, "TensorRedispatchGenerated.h")
    
    with open(header_path, 'w') as f:
        f.write(generate_header(funcs))
        
    with open(cpp_path, 'w') as f:
        f.write(generate_cpp(funcs))
        
    with open(bindings_path, 'w') as f:
        f.write(generate_bindings(funcs))
        
    d_list = []
    if derivatives_map:
        for k, v in derivatives_map.items():
            d_list.append(v)
    
    with open(autograd_nodes_path, 'w') as f:
        f.write(generate_autograd_nodes(d_list, funcs))

    with open(tpx_ops_h_path, 'w') as f:
        f.write(generate_tpx_ops_h(funcs))

    with open(tpx_ops_cpp_path, 'w') as f:
        f.write(generate_tpx_ops_cpp(funcs))

    with open(redispatch_h_path, 'w') as f:
        f.write(generate_redispatch_header(funcs))

    with open(tpx_autograd_reg_path, 'w') as f:
        f.write(generate_autograd_registration(funcs))

    with open(autocast_reg_path, 'w') as f:
        f.write(generate_autocast_registration(funcs))

    print(f"Generated \"{args.out_dir}\"")

    if args.pyi_template and args.pyi_out:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dtype_header_path = os.path.join(script_dir, "../../p10/include/DType.h")
        
        pyi_content = generate_pyi(funcs, args.pyi_template, dtype_header_path)
        with open(args.pyi_out, 'w') as f:
            f.write(pyi_content)
        print(f"Generated \"{args.pyi_out}\"")

    if args.pkg_out:
        functional_out = os.path.join(args.pkg_out, "functional.py")
        functional_content = generate_functional_py(funcs)
        with open(functional_out, 'w') as f:
            f.write(functional_content)
        print(f"Generated \"{functional_out}\"")
        

if __name__ == "__main__":
    main()
