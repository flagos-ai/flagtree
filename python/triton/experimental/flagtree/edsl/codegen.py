import ast
from typing import Any, Dict


class EdslCodeGenerator(ast.NodeVisitor):

    def __init__(self, operands, builder, *args, lscope=None, gscope=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.operands = operands
        self.builder = builder
        self.lscope: Dict[str, Any] = {} if lscope is None else lscope
        self.gscope: Dict[str, Any] = {} if gscope is None else gscope
        self.func = None

    def lookup(self, name: str) -> Any:
        for scope in (self.lscope, self.gscope):
            if name in scope:
                return scope[name]
        return None
