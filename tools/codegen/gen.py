import yaml
import os
import re
import argparse
import copy

import codegen_utils

# Type mapping from YAML schema to C++ signature
TYPE_MAP = {
    'int64_t[]': 'const std::vector<int64_t>&',
    'Tensor[]': 'const std::vector<Tensor>&',
    'DType': 'DType',
    'Device': 'Device',
    'double': 'double',
    'bool': 'bool',
    'Scalar': 'Scalar',
    'Scalar?': 'std::optional<Scalar>',
    'Tensor': 'const Tensor&',
    'Tensor(a!)': 'Tensor&',
    'Scalar?': 'std::optional<Scalar>',
    'int64_t?': 'std::optional<int64_t>',
}

# Type mapping for DispatchStub template arguments
STUB_TYPE_MAP = {
    'int64_t[]': 'const std::vector<int64_t>&',
    'Tensor[]': 'const std::vector<Tensor>&',
    'DType': 'DType',
    'Device': 'Device',
    'double': 'double',
    'bool': 'bool',
    'Scalar': 'Scalar',
    'Scalar?': 'std::optional<Scalar>',
    'Tensor': 'const Tensor&',
    'Tensor(a!)': 'Tensor&',
    'Scalar?': 'std::optional<Scalar>',
    'int64_t?': 'std::optional<int64_t>',
}

# Type mapping for Python Interface (.pyi)
PYI_TYPE_MAP = {
    'int64_t[]': 'Sequence[int]',
    'Tensor[]': 'Sequence[TensorBase]',
    'DType': 'DType',
    'Device': 'Device',
    'double': 'float',
    'bool': 'bool',
    'Scalar': 'Scalar',
    'Scalar?': 'Scalar | None',
    'Tensor': 'TensorBase',
    'Tensor(a!)': 'TensorBase',
    'int64_t': 'int',
    'int64_t?': 'int | None',
}

def default_handler(type_str, default):
    if default == 'Float32': return 'DType::Float32'
    if default == 'CPU': return 'Device(DeviceType::CPU)'
    if default == 'Int64': return 'DType::Int64'
    if default == 'Undefined': return 'DType::Undefined'
    if default == 'None': return 'std::nullopt'
    if type_str == 'Scalar' and re.match(r'^-?\d+(\.\d+)?$', default):
        return f'Scalar({default})'
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
    return default

def parse_func(func_str):
    arg_type_map = TYPE_MAP.copy()
    
    f = codegen_utils.parse_func(func_str, arg_type_map, default_handler)
    
    if f['schema_return_type'] == 'Tensor':
        f['return_type'] = 'Tensor'
    if f['schema_return_type'] == 'Tensor(a!)':
        f['return_type'] = 'Tensor&'
    elif f['schema_return_type'] == 'Tensor[]':
        f['return_type'] = 'std::vector<Tensor>'
        
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
        if f['schema_return_type'] == 'Tensor': ret_type = 'TensorBase'
        if f['schema_return_type'] == 'Tensor(a!)': ret_type = 'TensorBase'
        if f['schema_return_type'] == 'Tensor[]': ret_type = 'list[TensorBase]'
        
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

def generate_autograd_nodes(derivatives, native_funcs):
    lines = []
    lines.append("// Generated by tools/codegen/gen.py")
    lines.append("#pragma once")
    lines.append("#include \"Node.h\"")
    lines.append("#include \"TPXTensor.h\"")
    lines.append("#include \"Scalar.h\"")
    lines.append("#include <vector>")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("namespace tensorplay {")
    lines.append("namespace tpx {")
    lines.append("")

    native_map = {f['func_name']: f for f in native_funcs}

    for d in derivatives:
        func_name = parse_func(d['name'])['func_name']
        if func_name not in native_map:
            continue
            
        native_f = native_map[func_name]
        clean_name = "".join(x.title() for x in func_name.replace('.', '_').split('_'))
        node_name = clean_name + "Backward"
        
        formulas = {}
        for arg in native_f['args']:
            if arg['name'] in d:
                formulas[arg['name']] = d[arg['name']]
        
        used_vars = set()
        for formula in formulas.values():
            words = re.findall(r'\b[a-zA-Z_]\w*\b', formula)
            for w in words:
                if w in ['grad', 'grad_output', 'neg', 'pow', 'sin', 'cos', 'exp', 'log', 'tanh', 't', 'mm', 'div', 'mul', 'add', 'sub']: continue
                used_vars.add(w)
        
        members = []
        for arg in native_f['args']:
            if arg['name'] in used_vars:
                cpp_type = "Tensor" if arg['type'] in ['Tensor', 'Tensor(a!)'] else arg['type']
                if cpp_type == 'Tensor': cpp_type = 'Tensor'
                if arg['type'] in ['Scalar', 'double', 'int64_t']:
                    cpp_type = arg['type']
                if arg['type'] == 'int64_t[]':
                    cpp_type = 'std::vector<int64_t>'
                members.append({'name': arg['name'], 'type': cpp_type})
        
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
        lines.append("        if (inputs.empty() || !inputs[0].defined()) return {Tensor(), Tensor()};")
        lines.append("        const Tensor& grad = inputs[0];")
        lines.append("")
        lines.append("        variable_list grads;")
        
        for arg in native_f['args']:
            if arg['type'] in ['Tensor', 'Tensor(a!)']:
                if arg['name'] in formulas:
                    formula = formulas[arg['name']]
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

def generate_cpp(funcs):
    lines = []
    lines.append("// Generated by tools/codegen/gen.py")
    lines.append("#include \"Tensor.h\"")
    lines.append("#include \"Dispatcher.h\"")
    lines.append("#include \"Exception.h\"")
    lines.append("#include \"DispatchKey.h\"")
    lines.append("#include \"DType.h\"")
    lines.append("#include \"Scalar.h\"")
    lines.append("#include \"SizesAndStrides.h\"")
    lines.append("#include \"Device.h\"")
    lines.append("#include \"TypePromotion.h\"")
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
        
        lines.append(sig + " {")
        
        if f['name'] == 'copy_':
            lines.append('    if (!impl_ || !src.impl_) TP_THROW(RuntimeError, "Tensor not defined");')
            lines.append('    if (this->shape() != src.shape()) {')
            lines.append('        TP_THROW(RuntimeError, "copy_(): shapes mismatch (broadcasting not yet supported)");')
            lines.append('    }')
        
        # Pure Dispatch - No Autograd
        dispatch_key_source = "Device(DeviceType::CPU)"
        device_arg = next((arg for arg in f['args'] if arg['name'] == 'device'), None)
        if device_arg:
            dispatch_key_source = "device"
        elif f['variants'] == 'method':
            dispatch_key_source = "device()"
        
        lines.append(f"    DispatchKey key = computeDispatchKey({dispatch_key_source});")
        
        template_args = [f['return_type']]
        call_args = [f'"{f["func_name"]}"', "key"]
        for arg in f['args']:
            if arg['name'] == 'requires_grad': continue
            template_args.append(arg['stub_type'])
            if f['variants'] == 'method' and arg['name'] == 'self':
                call_args.append("*this")
            else:
                call_args.append(arg['name'])
                
        template_str = ", ".join(template_args)
        call_str = ", ".join(call_args)
        
        lines.append(f"    return DispatchStub<{template_str}>::call({call_str});")
        lines.append("}")
        lines.append("")
        
    lines.append("} // namespace tensorplay")
    return "\n".join(lines)

def generate_tpx_ops_h(funcs):
    lines = []
    lines.append("// Generated by tools/codegen/gen.py")
    lines.append("#pragma once")
    lines.append("#include \"TPXTensor.h\"")
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
    lines.append("#include \"TPXTensor.h\"")
    lines.append("#include \"Autograd.h\"")
    lines.append("#include \"tensorplay/ops/AutogradNodesGenerated.h\"")
    lines.append("#include \"Node.h\"")
    lines.append("#include \"tensorplay/ops/TPXOpsGenerated.h\"")
    lines.append("")
    lines.append("namespace tensorplay {")
    lines.append("namespace tpx {")
    lines.append("namespace ops {")
    lines.append("")
    lines.append("// Helper to convert Tensor list")
    lines.append("static std::vector<tensorplay::Tensor> to_core_list(const std::vector<tensorplay::tpx::Tensor>& list) {")
    lines.append("    std::vector<tensorplay::Tensor> core_list;")
    lines.append("    core_list.reserve(list.size());")
    lines.append("    for (const auto& t : list) {")
    lines.append("        core_list.push_back(t.core());")
    lines.append("    }")
    lines.append("    return core_list;")
    lines.append("}")
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
        
        # Autograd Setup
        if f.get('autograd'):
            lines.append('    bool requires_grad = false;')
            tensor_args_check = []
            for arg in f['args']:
                if arg['type'] in ['Tensor', 'Tensor(a!)']:
                    tensor_args_check.append(f"{arg['name']}.requires_grad()")
             
            if tensor_args_check:
                cond = " || ".join(tensor_args_check)
                lines.append(f"    if ({cond}) requires_grad = true;")
                 
            lines.append("    std::shared_ptr<Node> grad_fn;")
            lines.append("    if (requires_grad) {")
             
            node_cls = f.get('autograd_node_name', f['autograd'][0])
            autograd_args = f.get('autograd_args', [])
            
            node_arg_list = []
            if autograd_args:
                for arg_name in autograd_args:
                    node_arg_list.append(arg_name)
            
            node_args = ", ".join(node_arg_list)
            lines.append(f"        grad_fn = std::make_shared<{node_cls}>({node_args});")
            
            edge_args = []
            for arg in f['args']:
                if arg['type'] in ['Tensor', 'Tensor(a!)']:
                    edge_args.append(arg['name'])
            
            edge_args_str = ", ".join(edge_args)
            lines.append(f"        grad_fn->add_next_edge_list(collect_next_edges({edge_args_str}));")
            lines.append("    }")

        # Call underlying P10 Tensor method or function
        call_args = []
        for arg in f['args']:
            if f['variants'] == 'method' and arg['name'] == 'self':
                continue
            call_args.append(arg['name'])
        
        call_args_str = ", ".join(call_args)
        
        if f['variants'] == 'method':
            # self.core().method(...)
            # self is first arg
            self_arg = f['args'][0]['name']
            
            # For TPX wrapper, we call core().method()
            # arg.core() if it is a Tensor
            
            core_call_args = []
            for arg in f['args']:
                if arg['name'] == 'self': continue
                if arg['name'] == 'requires_grad': continue
                if arg['type'] in ['Tensor', 'Tensor(a!)']:
                    core_call_args.append(f"{arg['name']}.core()")
                elif arg['type'] == 'Tensor[]':
                    core_call_args.append(f"to_core_list({arg['name']})")
                else:
                    core_call_args.append(arg['name'])
            
            core_call_str = ", ".join(core_call_args)
            call_line = f"{self_arg}.core().{f['name']}({core_call_str})"
            
        else:
            # Tensor::function(...) -> tensorplay::Tensor::function(...)
            # Convert args to core tensors
            
            core_call_args = []
            for arg in f['args']:
                if arg['name'] == 'requires_grad': continue
                if arg['type'] in ['Tensor', 'Tensor(a!)']:
                    core_call_args.append(f"{arg['name']}.core()")
                elif arg['type'] == 'Tensor[]':
                    core_call_args.append(f"to_core_list({arg['name']})")
                else:
                    core_call_args.append(arg['name'])
            
            core_call_str = ", ".join(core_call_args)
            call_line = f"tensorplay::Tensor::{f['name']}({core_call_str})"

        if f['return_type'] == 'Tensor':
            lines.append(f"    Tensor result( {call_line} );")
            
            # Check for explicit requires_grad argument (for factories)
            has_requires_grad_arg = False
            for arg in f['args']:
                if arg['name'] == 'requires_grad':
                    has_requires_grad_arg = True
                    break
            
            if has_requires_grad_arg:
                lines.append("    result.set_requires_grad(requires_grad);")

            # Set history
            if f.get('autograd'):
                lines.append("    if (requires_grad && result.core().defined()) {")
                lines.append("        result.set_grad_fn(grad_fn);")
                lines.append("    }")
            
            lines.append("    return result;")
        elif f['return_type'] == 'std::vector<Tensor>':
             lines.append(f"    auto p10_result = {call_line};")
             lines.append("    std::vector<Tensor> result;")
             lines.append("    result.reserve(p10_result.size());")
             lines.append("    for (auto& t : p10_result) {")
             lines.append("        result.emplace_back(std::move(t));")
             lines.append("    }")
             lines.append("    return result;")
        elif f['return_type'] == 'Tensor&':
             lines.append(f"    {call_line};")
             first_arg = f['args'][0]['name']
             lines.append(f"    return {first_arg};")
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
    if val == 'None': return 'nb::none()'
    if val.startswith('{'):
        type_name = cpp_type.replace('const ', '').replace('&', '').strip()
        return f'{type_name}{val}'
    return val

def generate_bindings(funcs):
    lines = []
    lines.append("// Generated by tools/codegen/gen.py")
    lines.append("#pragma once")
    lines.append("#include <nanobind/nanobind.h>")
    lines.append("#include <nanobind/stl/string.h>")
    lines.append("#include <nanobind/stl/vector.h>")
    lines.append("#include <nanobind/stl/optional.h>")
    lines.append("#include \"TPXTensor.h\"")
    lines.append("#include \"tensorplay/ops/TPXOpsGenerated.h\"")
    lines.append("")
    lines.append("namespace nb = nanobind;")
    lines.append("")
    lines.append("namespace tensorplay {")
    lines.append("namespace python {")
    lines.append("")
    lines.append("using Tensor = tensorplay::tpx::Tensor;")
    lines.append("")
    lines.append("inline void bind_generated_tensor_methods(nb::class_<Tensor>& m) {")
    
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
                s = f'nb::arg("{arg_name}")'
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
    lines.append("inline void bind_generated_op_functions(nb::module_& m) {")
    
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
                s = f'nb::arg("{arg_name}")'
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
    lines.append("from ._C import *")
    lines.append("")
    
    seen_funcs = set()
    
    for f in funcs:
        if f['variants'] == 'method':
            name = f['name']
            if name in seen_funcs:
                continue
            seen_funcs.add(name)
            
            arg_strs = []
            call_args = []
            
            args = f['args']
            
            if not args or args[0]['name'] != 'self':
                continue
                
            arg_strs.append("input")
            
            for i in range(1, len(args)):
                arg = args[i]
                py_type = PYI_TYPE_MAP.get(arg['type'], arg['type'])
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
            lines.append(f"    return input.{name}({call_args_str})")
            lines.append("")
            
    return "\n".join(lines)

def generate_init_py(out_path, funcs):
    lines = []
    lines.append("# Copyright (c) 2025 zlx. All rights reserved.")
    lines.append("# Generated by tools/codegen/gen.py")
    lines.append("")
    lines.append('"""')
    lines.append("The tensorplay package offers a simple deep-learning framework")
    lines.append("designed for educational purposes and small-scale experiments.")
    lines.append("It defines a data structure for multidimensional arrays called Tensor,")
    lines.append("on which it encapsulates mathematical operations.")
    lines.append('"""')
    lines.append("")
    lines.append('__version__ = "1.0.0rc0"')
    lines.append("")
    lines.append("import os")
    lines.append("")
    lines.append("# Add the directory containing this file to DLL search path on Windows")
    lines.append("if os.name == 'nt':")
    lines.append("    try:")
    lines.append("        lib_path = os.path.join(os.path.dirname(__file__), 'lib')")
    lines.append("        if os.path.exists(lib_path):")
    lines.append("            os.add_dll_directory(lib_path)")
    lines.append("        else:")
    lines.append("            os.add_dll_directory(os.path.dirname(__file__))")
    lines.append("    except OSError:")
    lines.append("        pass")
    lines.append("")
    lines.append("from .tensor import Tensor")
    lines.append("from ._C import tensor, from_dlpack, to_dlpack")
    lines.append("from .scalar import Scalar")
    lines.append("from .device import Device, DeviceType")
    lines.append("")
    lines.append("device = Device")
    lines.append("")
    lines.append("from . import cuda")
    lines.append("from . import optim")
    lines.append("")
    lines.append("from .dtype import DType")
    lines.append("")
    lines.append("uint8 = DType.uint8")
    lines.append("int8 = DType.int8")
    lines.append("int16 = DType.int16")
    lines.append("uint16 = DType.uint16")
    lines.append("int32 = DType.int32")
    lines.append("int64 = DType.int64")
    lines.append("float32 = DType.float32")
    lines.append("float64 = DType.float64")
    lines.append("bool = DType.bool")
    lines.append("")
    
    lines.append("from ._C import (")
    factory_funcs = set()
    for f in funcs:
        if f['variants'] == 'function':
            factory_funcs.add(f['name'])
    sorted_funcs = sorted(list(factory_funcs))
    
    current_line = "    "
    for i, name in enumerate(sorted_funcs):
        if len(current_line) + len(name) + 2 > 75:
            lines.append(current_line)
            current_line = "    "
        current_line += name + ", "
    if current_line.strip():
        lines.append(current_line)
    lines.append(")")
    lines.append("")
    lines.append("from ._C import Size")
    lines.append("")
    lines.append("from .functional import *")
    lines.append("")
    lines.append("__all__ = [")
    lines.append('    "Tensor", "tensor", "from_dlpack", "Scalar", "Device", "DeviceType", "device",')
    lines.append('    "DType", "Size",')
    lines.append('    "uint8", "int8", "int16", "uint16", "int32", "int64", "float32", "float64", "bool",')
    
    current_line = "    "
    for i, name in enumerate(sorted_funcs):
        quoted_name = f'"{name}"'
        if len(current_line) + len(quoted_name) + 2 > 75:
            lines.append(current_line)
            current_line = "    "
        current_line += quoted_name + ", "
    if current_line.strip():
        lines.append(current_line)
    lines.append("]")
    lines.append("")
    lines.append("from . import __config__")
    
    with open(out_path, 'w') as f:
        f.write("\n".join(lines))

def generate_tpx_methods_decl(funcs):
    lines = []
    lines.append("// Generated by tools/codegen/gen.py")
    lines.append("// Included inside class Tensor")
    lines.append("")
    
    seen_sigs = set()
    
    for f in funcs:
        is_method = 'method' in f['variants']
        is_function = 'function' in f['variants']
        
        if is_method:
            sig = f['return_type'] + " " + f['name'] + "("
            arg_strs = []
            
            for arg in f['args']:
                if arg['name'] == 'self': continue
                s = f"{arg['cpp_type']} {arg['name']}"
                if arg['default']:
                    s += f" = {arg['default']}"
                arg_strs.append(s)
            
            sig += ", ".join(arg_strs) + ")"
            if '!' not in f['name'] and not any('!' in a['type'] for a in f['args']):
                 sig += " const"
            
            dedup_key = "method:" + sig
            if dedup_key in seen_sigs: continue
            seen_sigs.add(dedup_key)

            lines.append(f"{sig};")
            lines.append("")
            
        if is_function and not is_method:
             first_is_tensor = f['args'] and f['args'][0]['type'] in ['Tensor', 'Tensor(a!)']
             if not first_is_tensor:
                 sig = "static " + f['return_type'] + " " + f['name'] + "("
                 arg_strs = []
                 for arg in f['args']:
                     s = f"{arg['cpp_type']} {arg['name']}"
                     if arg['default']:
                         s += f" = {arg['default']}"
                     arg_strs.append(s)
                 
                 sig += ", ".join(arg_strs) + ")"
                 
                 dedup_key = "static:" + sig
                 if dedup_key in seen_sigs: continue
                 seen_sigs.add(dedup_key)

                 lines.append(f"{sig};")
                 lines.append("")
    return "\n".join(lines)

def generate_tpx_methods_impl(funcs):
    lines = []
    lines.append("// Generated by tools/codegen/gen.py")
    lines.append("// Included at end of TPXTensor.h")
    lines.append("")
    lines.append("namespace tensorplay { namespace tpx {")
    lines.append("")
    
    seen_sigs = set()
    
    for f in funcs:
        is_method = 'method' in f['variants']
        is_function = 'function' in f['variants']
        
        if is_method:
            sig = f['return_type'] + " Tensor::" + f['name'] + "("
            arg_strs = []
            call_args = []
            
            for arg in f['args']:
                if arg['name'] == 'self':
                    call_args.append("*this")
                    continue
                
                s = f"{arg['cpp_type']} {arg['name']}"
                arg_strs.append(s)
                call_args.append(arg['name'])
            
            sig += ", ".join(arg_strs) + ")"
            if '!' not in f['name'] and not any('!' in a['type'] for a in f['args']):
                 sig += " const"
            
            dedup_key = "method:" + sig
            if dedup_key in seen_sigs: continue
            seen_sigs.add(dedup_key)

            lines.append(f"inline {sig} {{")
            lines.append(f"    return ops::{f['name']}({', '.join(call_args)});")
            lines.append("}")
            lines.append("")
            
        if is_function and not is_method:
             first_is_tensor = f['args'] and f['args'][0]['type'] in ['Tensor', 'Tensor(a!)']
             if not first_is_tensor:
                 sig = f['return_type'] + " Tensor::" + f['name'] + "("
                 arg_strs = []
                 call_args = []
                 for arg in f['args']:
                     s = f"{arg['cpp_type']} {arg['name']}"
                     arg_strs.append(s)
                     call_args.append(arg['name'])
                 
                 sig += ", ".join(arg_strs) + ")"
                 
                 dedup_key = "static:" + sig
                 if dedup_key in seen_sigs: continue
                 seen_sigs.add(dedup_key)

                 lines.append(f"inline {sig} {{")
                 lines.append(f"    return ops::{f['name']}({', '.join(call_args)});")
                 lines.append("}")
                 lines.append("")

    lines.append("}} // namespace")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', required=True, help='Path to native_functions.yaml')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--pyi_template', help='Path to _C.pyi.in')
    parser.add_argument('--pyi_out', help='Path to output _C.pyi')
    parser.add_argument('--functional_out', help='Path to output functional.py')
    parser.add_argument('--derivatives', help='Path to derivatives.yaml')
    args = parser.parse_args()
    
    derivatives_map = {}
    if args.derivatives and os.path.exists(args.derivatives):
        with open(args.derivatives, 'r') as f:
            d_data = yaml.safe_load(f)
            if d_data:
                for item in d_data:
                    try:
                        d_f = parse_func(item['name'])
                        derivatives_map[d_f['func_name']] = item
                    except:
                        pass
    
    with open(args.yaml, 'r') as f:
        data = yaml.safe_load(f)
        
    if data is None:
        data = []
        
    funcs = []
    for item in data:
        base_f = parse_func(item['func'])
        base_f['autograd'] = item.get('autograd')
        
        if base_f['func_name'] in derivatives_map:
            func_name = base_f['func_name']
            clean_name = "".join(x.title() for x in func_name.replace('.', '_').split('_'))
            node_name = clean_name + "Backward"
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
            
            autograd_args = []
            for arg in base_f['args']:
                if arg['name'] in used_vars:
                    arg_name = arg['name']
                    autograd_args.append(arg_name)
            
            base_f['autograd_args'] = autograd_args
                 
        base_f['dispatch'] = item.get('dispatch')
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

    tpx_methods_decl_path = os.path.join(args.out_dir, "TPXTensorMethodsDeclGenerated.h")
    tpx_methods_impl_path = os.path.join(args.out_dir, "TPXTensorMethodsImplGenerated.h")
    
    with open(tpx_methods_decl_path, 'w') as f:
        f.write(generate_tpx_methods_decl(funcs))
        
    with open(tpx_methods_impl_path, 'w') as f:
        f.write(generate_tpx_methods_impl(funcs))

    print(f"Generated files in {args.out_dir}")

    if args.pyi_template and args.pyi_out:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dtype_header_path = os.path.join(script_dir, "../../p10/include/DType.h")
        
        pyi_content = generate_pyi(funcs, args.pyi_template, dtype_header_path)
        with open(args.pyi_out, 'w') as f:
            f.write(pyi_content)
        print(f"Generated {args.pyi_out}")

    if args.functional_out:
        functional_content = generate_functional_py(funcs)
        with open(args.functional_out, 'w') as f:
            f.write(functional_content)
        print(f"Generated {args.functional_out}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    init_py_path = os.path.join(script_dir, "../../tensorplay/__init__.py")
    generate_init_py(init_py_path, funcs)
    print(f"Generated {init_py_path}")

if __name__ == "__main__":
    main()
