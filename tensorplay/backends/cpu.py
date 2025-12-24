import tensorplay
from typing import List, Any

class CPUBackend:
    """
    A robust, fallback backend that executes the graph node-by-node using TensorPlay's eager execution engine.
    This ensures 100% operator coverage and stability, serving as the baseline for correctness.
    """
    def __init__(self):
        pass

    def compile(self, graph, example_inputs):
        # For CPU backend, we essentially interpret the graph.
        # We can optimize this by "lowering" to a more efficient interpreter
        # or by fusing ops if we had a C++ fusion engine exposed.
        # For now, we return a function that runs the interpreter.
        
        # We need to capture the graph.
        # But wait, the compiled function signature is usually (*args).
        
        def compiled_fn(*args):
            return self._execute_graph(graph, args)
            
        return compiled_fn

    def _execute_graph(self, graph, args):
        # Naive Interpreter
        # TODO: Optimize this (e.g. topological sort is implicit in node list, 
        # but we should pre-compute execution schedule to avoid overhead)
        
        env = {}
        
        # Bind inputs
        # Assuming inputs are the first N placeholders in order
        input_nodes = [n for n in graph.nodes if n.op == 'placeholder']
        if len(args) != len(input_nodes):
             raise RuntimeError(f"Expected {len(input_nodes)} inputs, got {len(args)}")
             
        for node, arg in zip(input_nodes, args):
            env[node] = arg
            
        # Execute
        for node in graph.nodes:
            if node.op == 'placeholder':
                continue
            
            if node.op == 'call_function':
                func = node.target
                # Resolve args
                # Node args can be other Nodes or constants
                node_args = [env[arg] if isinstance(arg, type(node)) else arg for arg in node.args]
                node_kwargs = {k: (env[v] if isinstance(v, type(node)) else v) for k, v in node.kwargs.items()}
                
                # Run op
                # If func is a string (from some legacy trace), map it. 
                # But our industrial frontend uses actual functions.
                if callable(func):
                    env[node] = func(*node_args, **node_kwargs)
                else:
                    raise RuntimeError(f"Unknown function target: {func}")
                    
            elif node.op == 'call_method':
                method_name = node.target
                node_args = [env[arg] if isinstance(arg, type(node)) else arg for arg in node.args]
                node_kwargs = {k: (env[v] if isinstance(v, type(node)) else v) for k, v in node.kwargs.items()}
                
                # First arg is 'self'
                self_obj = node_args[0]
                args = node_args[1:]
                
                if hasattr(self_obj, method_name):
                    method = getattr(self_obj, method_name)
                    env[node] = method(*args, **node_kwargs)
                else:
                    raise RuntimeError(f"Object {self_obj} has no method {method_name}")

            elif node.op == 'output':
                # Return result
                # args[0] is the return value(s)
                ret_val = node.args[0]
                if isinstance(ret_val, (tuple, list)):
                    return tuple(env[x] if isinstance(x, type(node)) else x for x in ret_val)
                else:
                    return env[ret_val] if isinstance(ret_val, type(node)) else ret_val
                    
        return None
