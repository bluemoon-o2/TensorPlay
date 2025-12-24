import textwrap
import hashlib

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

class TritonBackend:
    def __init__(self):
        pass

    def compile(self, graph, example_inputs):
        if not HAS_TRITON:
            raise RuntimeError("Triton is not installed. Cannot use Triton backend.")
        
        # 1. Schedule (Fuse ops)
        # For element-wise, we assume a single kernel for the whole graph for now.
        # In a real compiler, we would partition the graph.
        scheduler = TritonScheduler(graph)
        scheduled_graph = scheduler.schedule()

        # 2. Codegen
        codegen = TritonCodegen(scheduled_graph)
        kernel_code = codegen.generate()
        
        # 3. Compile/Load
        # Use a unique hash for the kernel name to avoid collisions in cache
        kernel_hash = hashlib.sha256(kernel_code.encode()).hexdigest()[:8]
        kernel_name = f"triton_kernel_{kernel_hash}"
        
        # Replace the generic name with specific hash
        kernel_code = kernel_code.replace("def triton_kernel(", f"def {kernel_name}(")
        # Need to replace the call in launcher too, but launcher is part of string
        # The generate() method hardcodes "triton_kernel[grid](...)" in launcher
        # We need to replace that too
        kernel_code = kernel_code.replace("triton_kernel[grid]", f"{kernel_name}[grid]")
        
        namespace = {}
        # We need to pass triton and tl to the exec environment
        exec_globals = globals().copy()
        exec_globals['triton'] = triton
        exec_globals['tl'] = tl
        
        exec(kernel_code, exec_globals, namespace)
        kernel_launch_fn = namespace['kernel_launch']
        
        return kernel_launch_fn

class TritonScheduler:
    def __init__(self, graph):
        self.graph = graph

    def schedule(self):
        # Current assumption: The entire graph is a single pointwise fusion group.
        # This is valid for the current test cases (add, mul, sin, etc.)
        # A robust scheduler would traverse the graph and break it at reduction boundaries.
        return self.graph

class TritonCodegen:
    """
    Generates Triton kernels aligned with PyTorch Inductor's style:
    - Explicit variable naming (in_ptr0, tmp0, out_ptr0)
    - XBLOCK based tiling
    - Autotuning configs
    - Robust masking
    """
    def __init__(self, graph):
        self.graph = graph
        self.inputs = [n for n in graph.nodes if n.op == 'placeholder']
        self.outputs = [n for n in graph.nodes if n.op == 'output']
        # Filter out inputs/outputs to get compute nodes
        self.computations = [n for n in graph.nodes if n.op not in ('placeholder', 'output')]
        
        self.var_map = {} # Map Node -> variable name (str)
        self.tmp_count = 0

    def _get_var(self, node):
        if node not in self.var_map:
            # Should not happen if topological sort is correct
            raise RuntimeError(f"Variable for node {node.name} not found")
        return self.var_map[node]

    def _new_tmp(self):
        name = f"tmp{self.tmp_count}"
        self.tmp_count += 1
        return name

    def generate(self):
        body_lines = []
        
        # 1. Preamble: Index calculation
        # xindex = xoffset + tl.arange(0, XBLOCK)[:]
        # xmask = xindex < xnumel
        body_lines.append("xoffset = tl.program_id(0) * XBLOCK")
        body_lines.append("xindex = xoffset + tl.arange(0, XBLOCK)[:]")
        body_lines.append("xmask = xindex < xnumel")
        
        # 2. Loads
        # tmp0 = tl.load(in_ptr0 + (xindex), xmask)
        for i, node in enumerate(self.inputs):
            var_name = self._new_tmp()
            self.var_map[node] = var_name
            # Assuming flat 1D layout for element-wise
            body_lines.append(f"{var_name} = tl.load(in_ptr{i} + (xindex), xmask)")

        # 3. Compute
        for node in self.computations:
            args_vars = [self._get_var(arg) for arg in node.args]
            res_var = self._new_tmp()
            self.var_map[node] = res_var
            
            op_code = self._gen_op(node.target, args_vars)
            body_lines.append(f"{res_var} = {op_code}")

        # 4. Stores
        # tl.store(out_ptr0 + (xindex), tmpX, xmask)
        # Assuming the output node contains the references to results
        # The 'output' node args are the actual result nodes
        output_args = self.outputs[0].args if self.outputs else []
        
        for i, res_node in enumerate(output_args):
            res_var = self._get_var(res_node)
            body_lines.append(f"tl.store(out_ptr{i} + (xindex), {res_var}, xmask)")

        # --- Construct Full Source ---
        
        src = "import triton\n"
        src += "import triton.language as tl\n\n"
        
        # Autotune Configs (Industrial grade efficiency)
        src += "@triton.autotune(\n"
        src += "    configs=[\n"
        src += "        triton.Config({'XBLOCK': 128}, num_warps=4),\n"
        src += "        triton.Config({'XBLOCK': 256}, num_warps=8),\n"
        src += "        triton.Config({'XBLOCK': 512}, num_warps=8),\n"
        src += "        triton.Config({'XBLOCK': 1024}, num_warps=8),\n"
        src += "    ],\n"
        src += "    key=['xnumel'],\n"
        src += ")\n"
        src += "@triton.jit\n"
        
        # Signature
        args = []
        for i in range(len(self.inputs)):
            args.append(f"in_ptr{i}")
        for i in range(len(output_args)):
            args.append(f"out_ptr{i}")
        args.append("xnumel")
        args.append("XBLOCK: tl.constexpr")
        
        src += f"def triton_kernel({', '.join(args)}):\n"
        src += textwrap.indent("\n".join(body_lines), "    ")
        src += "\n\n"
        
        # Launcher
        src += self._gen_launcher(len(self.inputs), len(output_args))
        
        return src

    def _gen_op(self, target, args):
        # target might be a function object, get its name or compare directly
        op_name = target.__name__ if hasattr(target, '__name__') else str(target)
        
        # Helper to check if target is a known tensorplay op
        # We need to import tensorplay, but doing it at top level might cause circular imports
        # So we do it inside method or use sys.modules
        import sys
        tp = sys.modules.get('tensorplay')
        
        # Check op names and target identity
        if op_name == 'add' or (tp and target == getattr(tp, 'add', None)): return f"{args[0]} + {args[1]}"
        if op_name == 'sub' or (tp and target == getattr(tp, 'sub', None)): return f"{args[0]} - {args[1]}"
        if op_name == 'mul' or (tp and target == getattr(tp, 'mul', None)): return f"{args[0]} * {args[1]}"
        if op_name == 'div' or (tp and target == getattr(tp, 'div', None)): return f"{args[0]} / {args[1]}"
        if op_name == 'sin' or (tp and target == getattr(tp, 'sin', None)): return f"tl.sin({args[0]})"
        if op_name == 'cos' or (tp and target == getattr(tp, 'cos', None)): return f"tl.cos({args[0]})"
        if op_name == 'exp' or (tp and target == getattr(tp, 'exp', None)): return f"tl.exp({args[0]})"
        if op_name == 'sqrt' or (tp and target == getattr(tp, 'sqrt', None)): return f"tl.sqrt({args[0]})"
        if op_name == 'abs' or (tp and target == getattr(tp, 'abs', None)): return f"tl.abs({args[0]})"
        if op_name == 'relu' or (tp and target == getattr(tp, 'relu', None)): return f"tl.maximum({args[0]}, 0.0)"
        if op_name == 'sigmoid' or (tp and target == getattr(tp, 'sigmoid', None)): return f"tl.sigmoid({args[0]})"
        
        raise NotImplementedError(f"Op {target} (name={op_name}) not supported")

    def _gen_launcher(self, num_inputs, num_outputs):
        body = []
        body.append("import tensorplay as tp")
        body.append("")
        body.append("xnumel = inputs[0].numel()")
        
        # Output allocation
        body.append("outputs = []")
        body.append(f"for _ in range({num_outputs}):")
        body.append("    outputs.append(tp.empty_like(inputs[0]))")
        
        body.append("")
        body.append("grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)")
        
        # Call kernel
        call_args = []
        for i in range(num_inputs):
            call_args.append(f"inputs[{i}]")
        for i in range(num_outputs):
            call_args.append(f"outputs[{i}]")
        call_args.append("xnumel")
        # XBLOCK is handled by autotuner/config
        
        body.append(f"triton_kernel[grid]({', '.join(call_args)})")
        body.append("return outputs[0] if len(outputs) == 1 else tuple(outputs)")
        
        src = "def kernel_launch(inputs):\n"
        src += textwrap.indent("\n".join(body), "    ")
        return src
