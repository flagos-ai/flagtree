from __future__ import annotations
import ast
import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from .cuda.codegen import EdslCUDACodeGenerator
from triton.language.core import TRITON_BUILTIN

from .dialect import EdslDialect

if TYPE_CHECKING:
    from .codegen import EdslCodeGenerator

T = TypeVar("T")


class UnknownDialectError(Exception):

    def __init__(self, dialect: Any, *args, **kwargs) -> None:
        super().__init__(f"the dialect {dialect} is unknown", *args, **kwargs)


class EdslFunction(Generic[T]):

    dialects: List[EdslDialect] = [EdslDialect("cuda", EdslCUDACodeGenerator)]

    def __init__(self, fn: T, dialect: Union[str, EdslDialect], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn: T = fn
        if isinstance(dialect, str):
            self.dialect = self.get_dialect(dialect)
        elif issubclass(type(dialect), EdslDialect):
            self.dialect = dialect
        else:
            self.dialect = None
        if self.dialect is None:
            raise UnknownDialectError(dialect)

    def get_func(self, operands, builder) -> Any:
        generator: EdslCodeGenerator = self.dialect.codegen_cls(operands, builder, gscope=self.__globals__)
        src: str = inspect.getsource(self.fn)
        module: ast.Module = ast.parse(src)
        generator.visit(module)
        return generator.func

    @staticmethod
    def get_dialect(name: str) -> Optional[EdslDialect]:
        for dialect in EdslFunction.dialects:
            if dialect.name == name:
                return dialect
        return None

    @property
    def __globals__(self) -> Dict[str, Any]:
        return self.fn.__globals__


def dialect(*, name: str) -> Callable[[T], EdslFunction[T]]:

    def decorator(fn: T) -> EdslFunction[T]:
        edsl: EdslFunction = EdslFunction(fn, name)
        setattr(edsl, TRITON_BUILTIN, True)
        return edsl

    return decorator
