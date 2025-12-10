
from __future__ import annotations
import torch

def iter_modules(module: torch.nn.Module, prefix: str = ""):
    yield prefix.rstrip("."), module
    for name, child in module.named_children():
        child_prefix = f"{prefix}{name}."
        yield from iter_modules(child, child_prefix)

def print_module_tree(module: torch.nn.Module, max_depth: int = 5):
    for path, mod in iter_modules(module):
        depth = 0 if path == "" else path.count(".")
        if depth > max_depth:
            continue
        cls = mod.__class__.__name__
        indent = "  " * depth
        print(f"{indent}{path or '<root>'} : {cls}")
