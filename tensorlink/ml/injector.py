import types
from typing import List, Dict, Set
import textwrap
import ast
import inspect


class VariableUsageAnalyzer(ast.NodeVisitor):
    """Analyzes variable reads and writes within a code block"""

    def __init__(self):
        self.variables_read = set()
        self.variables_written = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.variables_read.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.variables_written.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Track attribute access like self.x or decoder_layer.attention_type"""
        if isinstance(node.value, ast.Name):
            full_name = f"{node.value.id}.{node.attr}"
            if isinstance(node.ctx, ast.Load):
                self.variables_read.add(full_name)
            elif isinstance(node.ctx, ast.Store):
                self.variables_written.add(full_name)
        self.generic_visit(node)


class LoopFinder(ast.NodeVisitor):
    """Find the layer iteration loop"""

    def __init__(self):
        self.loop_node = None

    def visit_For(self, node):
        """Find loop that iterates over layers"""
        # Look for patterns like: for ... in self.layers
        if isinstance(node.iter, ast.Attribute):
            if node.iter.attr in ['layers', 'h', 'blocks', 'decoder_layers']:
                self.loop_node = node
                return

        # Look for: for ... in self.layers[:...]
        if isinstance(node.iter, ast.Subscript):
            if isinstance(node.iter.value, ast.Attribute):
                if node.iter.value.attr in ['layers', 'h', 'blocks', 'decoder_layers']:
                    self.loop_node = node
                    return

        self.generic_visit(node)


class LayerCallExtractor(ast.NodeVisitor):
    """Extract the layer call from loop body"""

    def __init__(self):
        self.layer_calls = []

    def visit_Assign(self, node):
        """Find assignments that call the layer"""
        if isinstance(node.value, ast.Call):
            # Get the variable being assigned to (e.g., hidden_states)
            if isinstance(node.targets[0], ast.Name):
                assigned_to = node.targets[0].id
            else:
                assigned_to = ast.unparse(node.targets[0])

            # Extract call arguments
            call_info = self._extract_call_info(node.value)
            call_info['assigned_to'] = assigned_to

            self.layer_calls.append(call_info)

        self.generic_visit(node)

    def _extract_call_info(self, call_node: ast.Call) -> Dict:
        """Extract arguments from a function call"""
        # Get positional arguments
        positional_args = [ast.unparse(arg) for arg in call_node.args]

        # Get keyword arguments
        keyword_args = {}
        has_var_kwargs = False

        for keyword in call_node.keywords:
            if keyword.arg is None:  # **kwargs
                has_var_kwargs = True
            else:
                keyword_args[keyword.arg] = ast.unparse(keyword.value)

        return {
            'args': positional_args,
            'kwargs': keyword_args,
            'has_var_kwargs': has_var_kwargs,
        }


class FunctionArgExtractor(ast.NodeVisitor):
    """Extract function arguments"""

    def __init__(self):
        self.args = set()

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            if arg.arg != 'self':
                self.args.add(arg.arg)


def _determine_loop_variables(
    var_analyzer: VariableUsageAnalyzer, loop_node: ast.For, function_args: Set[str]
) -> Dict:
    """
    Determine which variables need to be passed to workers and returned.

    Only includes variables that:
    - Exist before the loop (simple names only)
    - Don't reference loop iterator or its attributes
    """
    # Get iterator variable (e.g., 'decoder_layer' in 'for decoder_layer in self.decoder_layers')
    if isinstance(loop_node.target, ast.Name):
        iterator_var = loop_node.target.id
    else:
        iterator_var = ast.unparse(loop_node.target)

    # Filter out:
    # 1. Iterator variable and its attributes (decoder_layer, decoder_layer.attention_type)
    # 2. self.* attributes (module state)
    # 3. Any dotted names (only keep simple variable names)
    variables_read = set()
    for v in var_analyzer.variables_read:
        # Skip iterator and its attributes
        if v == iterator_var or v.startswith(f"{iterator_var}."):
            continue
        # Skip self.* attributes
        if v.startswith('self.'):
            continue
        # Only include simple variable names (no dots)
        if '.' not in v:
            variables_read.add(v)

    variables_written = set()
    for v in var_analyzer.variables_written:
        # Skip iterator
        if v == iterator_var or v.startswith(f"{iterator_var}."):
            continue
        # Skip self.*
        if v.startswith('self.'):
            continue
        # Only simple names
        if '.' not in v:
            variables_written.add(v)

    # Passthrough: variables both read and written (state that accumulates)
    passthrough_vars = variables_read & variables_written

    # Input-only: read but never written
    input_only_vars = variables_read - variables_written

    # Output-only: written but never read (rare)
    output_only_vars = variables_written - variables_read

    # Filter inputs to only include function arguments or passthrough vars
    # This excludes loop-created variables like 'layer_outputs'
    valid_input_only = input_only_vars & function_args

    return {
        'input_vars': valid_input_only,
        'passthrough_vars': passthrough_vars,
        'output_vars': output_only_vars,
        'all_inputs': valid_input_only | passthrough_vars,
        'all_outputs': output_only_vars | passthrough_vars,
    }


def generate_new_forward_method(
    parent_module, offloaded_modules: List
) -> types.FunctionType:
    """
    Generate a new forward method with loop replaced by worker calls.

    Args:
        parent_module: The module whose forward pass contains the loop
        offloaded_modules: List of OffloadedModule instances to call sequentially
        original_globals: Global namespace to use for the new function (optional)

    Returns:
        New forward function (unbound)
    """
    # Get original forward source
    original_forward = parent_module.forward
    source = inspect.getsource(original_forward)
    source = textwrap.dedent(source)

    # Parse and analyze
    tree = ast.parse(source)

    # Extract function arguments
    arg_extractor = FunctionArgExtractor()
    arg_extractor.visit(tree)

    # Find the loop
    loop_finder = LoopFinder()
    loop_finder.visit(tree)

    if not loop_finder.loop_node:
        raise ValueError("No suitable loop found in forward pass")

    # Analyze variable usage in loop
    var_analyzer = VariableUsageAnalyzer()
    for stmt in loop_finder.loop_node.body:
        var_analyzer.visit(stmt)

    # Extract the layer call to preserve kwargs
    layer_call_info = _extract_layer_call(loop_finder.loop_node)

    # Determine input and output variables
    loop_vars = _determine_loop_variables(
        var_analyzer, loop_finder.loop_node, arg_extractor.args
    )

    # Generate new forward code
    new_forward_code = _generate_new_forward_source(
        source, loop_finder.loop_node, layer_call_info, loop_vars, offloaded_modules
    )

    # Prepare namespace
    namespace = _get_model_module_globals(parent_module, original_forward)

    try:
        exec(new_forward_code, namespace)
        return namespace['forward']
    except Exception as e:
        print("=" * 80)
        print("ERROR COMPILING NEW FORWARD")
        print("=" * 80)
        print(new_forward_code)
        print("=" * 80)
        raise RuntimeError(f"Failed to compile: {e}")


def _extract_layer_call(loop_node: ast.For) -> Dict:
    """Extract the exact layer call signature from the loop body"""
    call_extractor = LayerCallExtractor()
    for stmt in loop_node.body:
        call_extractor.visit(stmt)

    if not call_extractor.layer_calls:
        raise ValueError("Could not find layer call in loop body")

    return call_extractor.layer_calls[0]


def _generate_new_forward_source(
    original_source: str,
    loop_node: ast.For,
    layer_call_info: Dict,
    loop_vars: Dict,
    offloaded_modules: List,
) -> str:
    """Generate new forward source with loop replaced by worker calls"""
    lines = original_source.split('\n')

    # Extract original function node to determine signature span
    module_ast = ast.parse(original_source)
    func_node = next(
        (
            n
            for n in module_ast.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward"
        ),
        None,
    )

    if func_node is None:
        raise RuntimeError("Could not locate original forward()")

    # Find where the function signature starts and ends in the original source
    func_start_line = func_node.lineno - 1  # 0-indexed
    func_body_start_line = func_node.body[0].lineno - 1  # First statement in body

    # Manually construct clean signature without type hints
    clean_sig = _manually_construct_signature(func_node)

    # Extract sections
    before_func = lines[:func_start_line]

    # Skip decorators in before_func
    clean_before_func = []
    for line in before_func:
        if not line.strip().startswith('@'):
            clean_before_func.append(line)

    # Get the body (starting from first statement, skipping original signature)
    func_body = lines[func_body_start_line : loop_node.lineno - 1]

    after_loop = (
        lines[loop_node.end_lineno :] if loop_node.end_lineno < len(lines) else []
    )

    # Get indentation from the first body line
    if func_body:
        first_body_line = func_body[0]
        indent = len(first_body_line) - len(first_body_line.lstrip())
    else:
        # Fallback to loop indentation
        loop_line = lines[loop_node.lineno - 1]
        indent = len(loop_line) - len(loop_line.lstrip())

    indent_str = ' ' * indent

    # Generate worker calls
    worker_calls = _generate_worker_calls(
        layer_call_info, loop_vars, indent_str, offloaded_modules
    )

    # Combine
    new_lines = []
    new_lines.extend(clean_before_func)

    # Add the clean signature (with proper indentation if needed)
    func_indent = ' ' * (
        len(lines[func_start_line]) - len(lines[func_start_line].lstrip())
    )
    new_lines.append(func_indent + clean_sig)

    # Add the function body before the loop
    new_lines.extend(func_body)

    # Add worker calls
    new_lines.append(f"{indent_str}# Offloaded layer groups")
    new_lines.extend(worker_calls)

    # Add everything after the loop
    new_lines.extend(after_loop)

    return '\n'.join(new_lines)


def _manually_construct_signature(func_node: ast.FunctionDef) -> str:
    """Manually construct a clean function signature without type hints"""
    args_parts = []

    # Regular args with defaults
    num_defaults = len(func_node.args.defaults)
    num_args = len(func_node.args.args)

    for i, arg in enumerate(func_node.args.args):
        arg_str = arg.arg
        # Defaults align to the end of args list
        default_idx = i - (num_args - num_defaults)
        if default_idx >= 0:
            default = func_node.args.defaults[default_idx]
            arg_str += f"={ast.unparse(default)}"
        args_parts.append(arg_str)

    # *args
    if func_node.args.vararg:
        args_parts.append(f"*{func_node.args.vararg.arg}")

    # Keyword-only args
    for arg, default in zip(func_node.args.kwonlyargs, func_node.args.kw_defaults):
        arg_str = arg.arg
        if default is not None:
            arg_str += f"={ast.unparse(default)}"
        args_parts.append(arg_str)

    # **kwargs
    if func_node.args.kwarg:
        args_parts.append(f"**{func_node.args.kwarg.arg}")

    return f"def {func_node.name}({', '.join(args_parts)}):"


def _generate_worker_calls(
    layer_call_info: Dict, loop_vars: Dict, indent: str, offloaded_modules: List
) -> List[str]:
    """Generate worker calls with proper variable passing and unpacking"""
    calls = []

    all_inputs = sorted(loop_vars['all_inputs'])
    all_outputs = sorted(loop_vars['all_outputs'])

    for idx, offloaded_module in enumerate(offloaded_modules):
        layer_range = getattr(offloaded_module, 'layer_range', 'unknown')

        # Comment
        calls.append(f"{indent}# Worker {idx}: layers {layer_range}")

        # Build the call
        call_str = f"{indent}"

        # Handle outputs (unpack if multiple)
        if len(all_outputs) > 1:
            outputs_str = ", ".join(all_outputs)
            call_str += f"{outputs_str} = self.offloaded_modules[{idx}]("
        elif len(all_outputs) == 1:
            call_str += f"{list(all_outputs)[0]} = self.offloaded_modules[{idx}]("
        else:
            call_str += f"self.offloaded_modules[{idx}]("

        # Add all input variables as arguments
        arg_parts = []
        for var in all_inputs:
            arg_parts.append(f"{indent}    {var}")

        # Add keyword arguments from original call that aren't already in inputs
        for kw_arg, kw_value in layer_call_info['kwargs'].items():
            if kw_arg not in all_inputs:
                arg_parts.append(f"{indent}    {kw_arg}={kw_value}")

        # Add **kwargs if present in original
        if layer_call_info['has_var_kwargs']:
            arg_parts.append(f"{indent}    **flash_attn_kwargs")

        if arg_parts:
            call_str += "\n" + ",\n".join(arg_parts) + f"\n{indent}"

        call_str += ")"

        calls.append(call_str)

    return calls


def get_loop_io_signature(parent_module) -> Dict:
    """
    Analyze the forward pass loop and return expected inputs/outputs without injecting.

    Returns a dictionary with:
    - 'input_vars': Variables read but not written (function args only)
    - 'passthrough_vars': Variables both read and written (accumulating state)
    - 'output_vars': Variables written but not read
    - 'all_inputs': All variables that should be passed in
    - 'all_outputs': All variables that should be returned
    - 'layer_call_info': Information about the original layer call signature

    Args:
        parent_module: The module whose forward pass contains the loop

    Returns:
        Dict containing input/output variable information
    """
    # Get original forward source
    original_forward = parent_module.forward
    source = inspect.getsource(original_forward)
    source = textwrap.dedent(source)

    # Parse and analyze
    tree = ast.parse(source)

    # Extract function arguments
    arg_extractor = FunctionArgExtractor()
    arg_extractor.visit(tree)

    # Find the loop
    loop_finder = LoopFinder()
    loop_finder.visit(tree)

    if not loop_finder.loop_node:
        raise ValueError("No suitable loop found in forward pass")

    # Analyze variable usage in loop
    var_analyzer = VariableUsageAnalyzer()
    for stmt in loop_finder.loop_node.body:
        var_analyzer.visit(stmt)

    # Extract the layer call to preserve kwargs
    call_extractor = LayerCallExtractor()
    for stmt in loop_finder.loop_node.body:
        call_extractor.visit(stmt)

    layer_call_info = (
        call_extractor.layer_calls[0] if call_extractor.layer_calls else {}
    )

    # Determine input and output variables
    loop_vars = _determine_loop_variables(
        var_analyzer, loop_finder.loop_node, arg_extractor.args
    )

    # Add layer call info
    result = {**loop_vars, 'layer_call_info': layer_call_info}

    return result


def _get_model_module_globals(parent_module, original_forward) -> dict:
    """
    Get the correct global namespace for the model's forward method.

    This handles cases where the forward method might be wrapped/decorated
    and its __globals__ points to the wrong module.
    """
    import sys

    # Attempt 1: Get from the parent module's class
    parent_class = type(parent_module)
    class_module_name = parent_class.__module__

    if class_module_name in sys.modules:
        class_module = sys.modules[class_module_name]
        return vars(class_module)

    # Attempt 2: Try to get from the forward method's __globals__
    if hasattr(original_forward, '__func__'):
        func_globals = original_forward.__func__.__globals__
    elif hasattr(original_forward, '__globals__'):
        func_globals = original_forward.__globals__
    else:
        func_globals = {}

    # Attempt 3: If the globals look wrong (like transformers.utils.generic),
    # try to import the actual model module
    if '__name__' in func_globals:
        globals_module = func_globals['__name__']
        if 'utils' in globals_module or 'generic' in globals_module:
            # This is a utility module, not the actual model module
            # Try to get the real module
            if class_module_name in sys.modules:
                return vars(sys.modules[class_module_name])

    return func_globals
