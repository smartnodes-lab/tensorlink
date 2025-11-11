"""Test the inference of a tiny model across two local worker nodes"""

import json

from tensorlink.ml.graphing import ModelParser

parser = ModelParser()
config = parser.create_distributed_config(
    "distilbert/distilgpt2",
    {
        '509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f': {
            'id': '509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f',
            'gpu_memory': 300000000.0,
            'total_gpu_memory': 300000000.0,
            'role': 'W',
            'training': False,
        },
        '209d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f': {
            'id': '509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f',
            'gpu_memory': 300000000.0,
            'total_gpu_memory': 300000000.0,
            'role': 'W',
            'training': False,
        },
    },
    False,
    False,
    False,
    False,
)


from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel
from typing import Dict, List, Optional, Callable
import textwrap
import inspect
import ast


class ForwardCheckpoint:
    """
    Represents a checkpoint in the forward pass execution.
    Stores the execution state at a specific point.
    """

    def __init__(self, name: str, position: int, is_module_boundary: bool = False):
        self.name = name
        self.position = position  # Line number or execution order
        self.is_module_boundary = is_module_boundary
        self.module_id = None if not is_module_boundary else name


class ForwardExecutionPlan:
    """
    Analyzes a forward method and creates an execution plan with checkpoints
    at module boundaries.
    """

    def __init__(self, forward_method: Callable, module_names: List[str]):
        """
        Args:
            forward_method: The forward() method to analyze
            module_names: List of module attribute names (e.g., ['wte', 'wpe', 'h', 'ln_f'])
        """
        self.forward_method = forward_method
        self.module_names = set(module_names)
        self.checkpoints: List[ForwardCheckpoint] = []
        self.execution_segments: List[Dict] = []

        self._analyze_forward_method()

    def _analyze_forward_method(self):
        """
        Parse the forward method to identify module usage points.
        """
        # Get the source code of the forward method
        source = inspect.getsource(self.forward_method)
        source = textwrap.dedent(source)

        # Parse into AST
        tree = ast.parse(source)

        # Find all attribute accesses and calls
        module_usages = []

        class ModuleUsageVisitor(ast.NodeVisitor):
            def __init__(self, module_names):
                self.module_names = module_names
                self.usages = []
                self.current_line = 0

            def visit_Call(self, node):
                # Check if this is a call to a module (e.g., self.wte(input_ids))
                if isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == 'self'
                    ):
                        if node.func.attr in self.module_names:
                            self.usages.append(
                                {
                                    'module': node.func.attr,
                                    'line': node.lineno,
                                    'type': 'call',
                                }
                            )

                # Check for nested calls like self.h[i](hidden_states)
                if isinstance(node.func, ast.Subscript):
                    if isinstance(node.func.value, ast.Attribute):
                        if (
                            isinstance(node.func.value.value, ast.Name)
                            and node.func.value.value.id == 'self'
                        ):
                            if node.func.value.attr in self.module_names:
                                self.usages.append(
                                    {
                                        'module': node.func.value.attr,
                                        'line': node.lineno,
                                        'type': 'subscript_call',
                                        'is_loop': True,
                                    }
                                )

                self.generic_visit(node)

            def visit_For(self, node):
                # Track for loops that iterate over modules
                if isinstance(node.iter, ast.Call):
                    if (
                        isinstance(node.iter.func, ast.Name)
                        and node.iter.func.id == 'enumerate'
                    ):
                        if isinstance(node.iter.args[0], ast.Attribute):
                            if node.iter.args[0].attr in self.module_names:
                                self.usages.append(
                                    {
                                        'module': node.iter.args[0].attr,
                                        'line': node.lineno,
                                        'type': 'loop',
                                        'is_loop': True,
                                    }
                                )
                self.generic_visit(node)

        visitor = ModuleUsageVisitor(self.module_names)
        visitor.visit(tree)
        module_usages = visitor.usages

        # Sort by line number
        module_usages.sort(key=lambda x: x['line'])

        # Create checkpoints at module boundaries
        for i, usage in enumerate(module_usages):
            checkpoint = ForwardCheckpoint(
                name=usage['module'], position=usage['line'], is_module_boundary=True
            )
            checkpoint.module_id = usage['module']
            checkpoint.is_loop = usage.get('is_loop', False)
            self.checkpoints.append(checkpoint)

        # Create execution segments between checkpoints
        self._create_execution_segments()

    def _create_execution_segments(self):
        """
        Divide the forward method into segments that can be executed independently.
        """
        if not self.checkpoints:
            return

        # Segment 0: From start to first module
        self.execution_segments.append(
            {
                'segment_id': 0,
                'start_line': 0,
                'end_checkpoint': self.checkpoints[0],
                'type': 'pre_module',
                'executor': 'user',
            }
        )

        # Middle segments: Between modules
        for i in range(len(self.checkpoints) - 1):
            self.execution_segments.append(
                {
                    'segment_id': i + 1,
                    'start_checkpoint': self.checkpoints[i],
                    'end_checkpoint': self.checkpoints[i + 1],
                    'type': 'inter_module',
                    'executor': 'worker',
                }
            )

        # Final segment: After last module
        self.execution_segments.append(
            {
                'segment_id': len(self.checkpoints),
                'start_checkpoint': self.checkpoints[-1],
                'end_line': None,
                'type': 'post_module',
                'executor': 'user',
            }
        )

    def get_user_segments(self) -> List[Dict]:
        """Get segments to be executed on user side."""
        return [s for s in self.execution_segments if s['executor'] == 'user']

    def get_worker_segments(self) -> List[Dict]:
        """Get segments to be executed on worker side."""
        return [s for s in self.execution_segments if s['executor'] == 'worker']

    def get_first_worker_module(self) -> Optional[str]:
        """Get the name of the first module executed on worker."""
        if not self.checkpoints:
            return None
        return self.checkpoints[0].module_id

    def get_last_worker_module(self) -> Optional[str]:
        """Get the name of the last module executed on worker."""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1].module_id


model_config = AutoConfig.from_pretrained("distilbert/distilgpt2")
with init_empty_weights():
    model_skeleton = AutoModel.from_config(model_config)

all_modules = [name for name, _ in model_skeleton.named_children()]
execution_plan = ForwardExecutionPlan(model_skeleton.forward, all_modules)
