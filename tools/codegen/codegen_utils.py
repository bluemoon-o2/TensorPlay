import re

def parse_args(arg_str, type_map, default_handler=None):
    args = []
    if not arg_str:
        return args
    
    # Split by comma, respecting braces/parens
    parts = []
    current_part = []
    depth = 0
    for char in arg_str:
        if char in '({[':
            depth += 1
            current_part.append(char)
        elif char in ')}]':
            depth -= 1
            current_part.append(char)
        elif char == ',' and depth == 0:
            parts.append("".join(current_part).strip())
            current_part = []
        else:
            current_part.append(char)
    if current_part:
        parts.append("".join(current_part).strip())

    for p in parts:
        # Handle kwonly marker '*'
        if p == '*':
            continue

        # Parse "Type name=Default"
        default = None
        if '=' in p:
            p, default = p.split('=', 1)
            p = p.strip()
            default = default.strip()
        
        # Parse "Type name"
        if ' ' not in p:
            raise ValueError(f"Invalid arg: {p}")
        
        type_str, name = p.rsplit(' ', 1)
        type_str = type_str.strip()
        name = name.strip()
        
        cpp_type = type_map.get(type_str, type_str)

        if default and default_handler:
            default = default_handler(type_str, default)
        
        args.append({
            'name': name,
            'type': type_str,
            'cpp_type': cpp_type,
            'default': default
        })
    return args

def parse_func(func_str, type_map, default_handler=None):
    # "name(args) -> Return"
    # Allow dot in name for overloads (e.g. add.Tensor)
    match = re.match(r'([\w\.]+)\((.*)\)\s*->\s*(.*)', func_str)
    if not match:
        raise ValueError(f"Invalid func: {func_str}")
    
    full_name = match.group(1)
    # Strip overload name for C++ function name
    name = full_name.split('.')[0]
    
    args_str = match.group(2)
    return_type_str = match.group(3).strip()
    
    return_type = type_map.get(return_type_str, return_type_str)
        
    args = parse_args(args_str, type_map, default_handler)
    
    return {
        'name': name,
        'func_name': full_name, # Original name in schema
        'args': args,
        'return_type': return_type,
        'schema_return_type': return_type_str
    }
