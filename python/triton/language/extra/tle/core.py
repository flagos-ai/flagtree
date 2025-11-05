import numpy as np
import triton.language.core as tl
from typing import Optional, Tuple, overload
from . import types as tle

from triton.language import semantic as tl_semantic
from triton.language.core import (
    _tensor_member_fn,
    _shape_check_impl,
    _unwrap_if_constexpr,
    builtin,
    constexpr,
    tensor,
    range,
)
class pipeline(range):
    """
    Iterator that counts upward forever, with parallel execution semantics.

    This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
    :code:`triton.jit` functions. In addition, it allows user to pass extra attributes to the compiler.
    :param bind_sub_block: Tells the compiler if multiple vector cores participate in the loop.
        This is used in the mixed cube-vector kernel on 910B. The number of vector cores is determined by the number of
        iteration in this loop. Currently on 910B, max 2 vector cores could be used.
    """
    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None):
        super().__init__(arg1, arg2, step, num_stages, loop_unroll_factor)


@tl.builtin
def alloc(
    shape: tuple,
    dtype: tl.dtype,
    layout: Optional[tle.shared_layout_encoding] = None,
    scope: tle.storage_kind = tle.storage_kind.smem,
    _semantic=None,
) -> tle.buffered_tensor:
    # Map scope to storage for backward compatibility
    storage = scope

    unwrapped_shape = [tl._unwrap_if_constexpr(dim) for dim in shape]
    full_shape =  unwrapped_shape
    dtype = tl._unwrap_if_constexpr(dtype)
    elem_type = dtype.to_ir(_semantic.builder)
    # Unwrap layout if it's a constexpr
    layout = tl._unwrap_if_constexpr(layout)

    if layout is None:
        if storage == tle.storage_kind.smem:
            layout = tle.swizzled_shared_layout_encoding.make_default(rank=len(shape))
            layout_handle = _semantic.builder.make_swizzled_shared_encoding_attr(
                layout.vectorSize,
                layout.perPhase,
                layout.maxPhase,
                layout.order,
                layout.numCTAsPerCGA,
                layout.numCTASplit,
                layout.numCTAOrder,
            )
        else:
            layout = tle.tensor_memory_layout_encoding.make_default(shape)
            layout_handle = _semantic.builder.make_tensor_memory_encoding_attr(
                layout.blockM,
                layout.blockN,
                layout.unpacked,
                layout.CTASplitM,
                layout.CTASplitN,
            )
    else:
        # Use the provided layout
        layout_handle = layout.to_ir(_semantic.builder)
    if storage == tle.storage_kind.smem:
        tensor_handle = _semantic.builder.create_local_alloc(full_shape, elem_type, layout_handle)
    else:
        raise ValueError("tmem not support for now")

    return tle.buffered_tensor(tensor_handle, dtype, unwrapped_shape, storage, layout, _semantic)



@tl.builtin
def copy(
    src: tl.tensor,
    result: tle.buffered_tensor,
    shape: tuple,
    _semantic=None,
) -> tle.buffered_tensor:
    """
    copy data from global to local memory asynchronously.
    """
    mask=None
    other=None
    boundary_check=()
    padding_option=""
    cache_modifier=""
    eviction_policy=""
    volatile=False
    tt_load  = _semantic.load(src, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy,
                          volatile, None)
    block_type = tl.block_type(tt_load.type.element_ty, src.type.shape)
    shared_type = result.type
    tt_local_alloc = _semantic.builder.create_local_alloc(shared_type.to_ir(_semantic.builder), tt_load.handle)
    assert result.real_handle is None, "buffer already initialized, for now shared memory buffer can only be initialized once no reuse"
    result.real_handle = tt_local_alloc
    return

@tl.builtin
def local_load(
    buffer: tle.buffered_tensor,
    _semantic=None,
) -> tle.buffered_tensor:

    block_type = tl.block_type(buffer.type.element_ty, buffer.type.shape)
    output = _semantic.builder.create_local_load(block_type.to_ir(_semantic.builder), buffer.real_handle)
    return tl.tensor(output, block_type)