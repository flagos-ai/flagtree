import builtins
import inspect
import math
import os
import sqlite3
import threading
import time
import weakref
import warnings
from collections import OrderedDict
from typing import Dict, Optional

import triton
from triton._C.libtriton import mlu

from .autotuner import Autotuner, Heuristics, Config
from .jit import JITFunction, KernelInterface, TensorWrapper


def decompose_tensor_wrapper(obj):
    # TODO: Currently fast_libentry only accepts inputs tensors as instances of `torch.Tensor`.
    # This function will decompose `torch.Tensor` out of triton's `TensorWrapper`.
    # It should be consitent with triton's implementation. When the data structure of
    # `TensorWrapper` changes, this code also needs to be modified.
    # Refers to:
    # https://github.com/triton-lang/triton/blob/main/python/triton/runime/jit.py#L934
    import torch
    if hasattr(obj, "data_ptr"):
        if isinstance(obj, torch.Tensor):
            return obj
        elif isinstance(obj, TensorWrapper):
            return obj.base
        else:
            raise TypeError(f"fast libentry only supports torch.Tensor and TensorWrapper, but got {type(obj).__name__}")
    return obj


class FastLibentry(KernelInterface):
    """ Fast mode of kernel interface with c++ host code.
    """

    def __init__(
        self,
        fn,
    ):
        self.fn = fn
        self.arg_names = fn.arg_names
        while not isinstance(fn, JITFunction):
            fn = fn.fn
        self.jit_function: JITFunction = fn

        self.kernel_runner = mlu.runner.KernelRunner(self.fn, self.jit_function)

    def run(self, *args, **kwargs):
        """ Entry point for kernel launch."""
        return self.kernel_runner.launch(*args, **kwargs)

    @staticmethod
    def compile_kernel(fn, *args, **kwargs):
        """ Compile kernel in a thread-safe way."""
        with threading.Lock():
            args = [decompose_tensor_wrapper(arg) for arg in args]
            kwargs.update({k: decompose_tensor_wrapper(v) for k, v in kwargs.items()})

            kernel = fn.run(*args, **kwargs)
            jit_function = None
            # collect constexpr arguments for grid computation
            constexprs = {}
            tune_constexprs = {}
            heur_constexprs = {}
            while not isinstance(fn, JITFunction):
                if isinstance(fn, Autotuner):
                    config = fn.best_config
                    constexprs["num_warps"] = config.num_warps
                    constexprs["num_stages"] = config.num_stages
                    constexprs["pipeline_strategies"] = config.pipeline_strategies
                    constexprs = {**constexprs, **config.kwargs}
                    tune_constexprs = {**tune_constexprs, **config.kwargs}
                elif isinstance(fn, Heuristics):
                    for v, heur in fn.values.items():
                        heur_constexprs[v] = heur({
                            **dict(zip(fn.arg_names, args)),
                            **kwargs,
                            **constexprs,
                        })
                        constexprs[v] = heur_constexprs[v]
                else:
                    raise RuntimeError("Invalid Runtime Function")
                fn = fn.fn

            if isinstance(fn, JITFunction):
                jit_function = fn
            if jit_function is None:
                raise RuntimeError("can't get JITFunction")
            for p in jit_function.params:
                if (p.is_constexpr and p.name not in constexprs and (p.default is not inspect._empty)):
                    constexprs[p.name] = p.default
            return (
                kernel,
                constexprs,
                tune_constexprs,
                heur_constexprs,
            )


def fast_libentry():
    """
    Decorator for fast triton library entries.
    """

    def decorator(fn):
        tmp_fn = fn
        while not isinstance(tmp_fn, JITFunction):
            if hasattr(tmp_fn, 'fn'):
                tmp_fn = tmp_fn.fn
            else:
                break
        if not isinstance(tmp_fn, JITFunction):
            warnings.warn(
                f'JITFunction not found in decorator chain, maybe your are in interpreted mode, fallback to {fn}')
            return fn

        return FastLibentry(fn)

    return decorator
