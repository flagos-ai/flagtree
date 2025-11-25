from __future__ import annotations
from typing import TYPE_CHECKING, Final, Type

if TYPE_CHECKING:
    from .codegen import EdslCodeGenerator


class EdslDialect(object):

    def __init__(self, name: str, codegen_cls: Type[EdslCodeGenerator], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name: Final[str] = name
        self.codegen_cls: Final[Type[EdslCodeGenerator]] = codegen_cls
