import ast
from typing import Any
from typing_extensions import override

from ..codegen import EdslCodeGenerator


class EdslCUDACodeGenerator(EdslCodeGenerator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def call_function(self, func: Any, *args, **kwargs) -> Any:
        return func(*args, **kwargs, _builder=self.builder)

    @override
    def visit_Assign(self, node: ast.Assign) -> None:
        # TODO: we only support single target assignment for now, and only single name in the target
        [target] = node.targets
        self.lscope[target.id] = self.visit(node.value)

    @override
    def visit_Attribute(self, node: ast.Attribute) -> Any:
        ret = self.visit(node.value)
        if ret is not None:
            ret = getattr(ret, node.attr)
        return ret

    @override
    def visit_Call(self, node: ast.Call) -> Any:
        fn = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        return self.call_function(fn, *args)

    @override
    def visit_Constant(self, node) -> Any:
        return node.value

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        args_len = len(node.args.args)
        operands_len = len(self.operands)
        assert args_len == operands_len, f"The signature of the caller and the callee don't match: {args_len} != {operands_len}"
        for arg, operand in zip(node.args.args, self.operands):
            self.lscope[arg.arg] = operand
        for stmt in node.body:
            self.visit(stmt)

    @override
    def visit_Module(self, node: ast.Module) -> Any:
        [fn] = node.body
        return self.visit(fn)

    @override
    def visit_Name(self, node: ast.Name) -> Any:
        return self.lookup(node.id)
