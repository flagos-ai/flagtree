import torch
import os

def is_do_bench_npu():
    enable_bench_npu = os.getenv("TRITON_BENCH_METHOD", 'default').lower() == 'npu'
    if torch.npu.is_available() and enable_bench_npu:
        return True
    return False


def collect_files(base_dir):
    import pandas as pd
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file != 'op_statistic.csv':
                continue
            target_file = os.path.join(root, file)
            df = pd.read_csv(target_file)
            triton_rows = df[df['OP Type'].str.startswith('triton', na=False)]
            if not triton_rows.empty:
                return triton_rows['Avg Time(us)'].values[0]
            return float('inf')
    return float('inf')


def collect_single(base_dir: str, key: str = None) -> float:
    if not os.path.exists(base_dir):
        return float('inf')

    import pandas as pd
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file != 'op_statistic.csv':
                continue
            target_file = os.path.join(root, file)
            df = pd.read_csv(target_file)
            if key is not None:
                key_rows = df[df['OP Type'].str.startswith(key, na=False)]
                if not key_rows.empty:
                    return key_rows['Avg Time(us)'].values[0]
                return float('inf')
            else:
                # default: read the first row except header
                return df.loc[0, 'Avg Time(us)']

    return float('inf')


def do_bench_npu(fn, warmup=5, active=30, prof_dir=None, keep_res=False):
    import torch_npu
    import multiprocessing
    from triton import runtime
    from datetime import datetime, timezone

    # warmup kernel
    fn()
    torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )
    skip_first = 1
    wait = 0
    repeat = 1
    total = skip_first + (wait + warmup + active) * repeat

    if prof_dir is not None:
        torch_path = prof_dir
    else:
        process = multiprocessing.current_process()
        pid = process.pid
        process_name = process.name
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(runtime.cache.get_home_dir(), ".triton", "profile_results")
        torch_path = os.path.join(base_path, f"prof_{timestamp}_{process_name}-{pid}")
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(total):
            fn()
            prof.step()
            torch.npu.synchronize()

    time = collect_single(torch_path)

    if not keep_res:
        import shutil
        if os.path.exists(torch_path):
            shutil.rmtree(torch_path)

    return time


def ext_do_bench_npu(fn, warmup, rep, quantiles, return_mode):
    import torch
    from triton.testing import _summarize_statistics
    avg_time = do_bench_npu(fn, warmup=max(5, warmup), active=max(30, rep))
    return _summarize_statistics(torch.tensor([avg_time], dtype=torch.float), quantiles, return_mode)


def patch_triton_language():
    # Patch the triton language API here because triton's __init__.py
    # import testing in the last stages.
    from triton.language.tensor_descriptor import (
        tensor_descriptor,
        tensor_descriptor_type,
    )

    from triton.language.core_ext import (
        dot,
        cast,
        gather,
        get_element,
        insert_slice,
        extract_slice,
        trans,
        __add__,
        __radd__,
        __sub__,
        __rsub__,
        __mul__,
        __rmul__,
        __lshift__,
        __rshift__,
        parallel,
        compile_hint,
        make_tensor_descriptor,
        load_tensor_descriptor,
        store_tensor_descriptor,
        multibuffer,
        sync_block_all,
        sync_block_set,
        sync_block_wait,
        dtype_to_ir,
        sort
    )
    from triton.language.standard_ext import flip, sigmoid, softmax, isfinited, finitef, rint, atan2
    from triton.language.math_ext import (
        umulhi,
        exp,
        exp2,
        log,
        log2,
        cos,
        sin,
        sqrt,
        sqrt_rn,
        rsqrt,
        div_rn,
        erf,
        tanh,
        floor,
        ceil,
        _check_dtype,
        fma,
    )
    from triton.language.semantic_ext import (
        arange,
        floordiv,
        atom_red_typechecking_impl,
        atomic_cas,
        atomic_max,
        atomic_min,
        _load_legacy,
        maximum,
        minimum,
        mod,
        invert,
        logical_and,
        logical_or,
        not_,
        and_,
        or_,
        xor_,
        minus,
        dot_scaled,
    )
    from triton import language

    language.cast = cast
    language.dot = dot
    language.flip = flip
    language.sigmoid = sigmoid
    language.softmax = softmax
    language.gather = gather
    language.insert_slice = insert_slice
    language.extract_slice = extract_slice
    language.get_element = get_element
    language.tensor.__add__ = __add__
    language.tensor.__radd__ = __radd__
    language.tensor.__sub__ = __sub__
    language.tensor.__rsub__ = __rsub__
    language.tensor.__mul__ = __mul__
    language.tensor.__rmul__ = __rmul__
    language.tensor.__lshift__ = __lshift__
    language.tensor.__rshift__ = __rshift__
    language.trans = trans
    language.parallel = parallel
    language.compile_hint = compile_hint
    language.sort = sort
    language.multibuffer = multibuffer
    language.sync_block_all = sync_block_all
    language.sync_block_set = sync_block_set
    language.sync_block_wait = sync_block_wait
    language.make_tensor_descriptor = make_tensor_descriptor
    language.tensor_descriptor = tensor_descriptor
    language.tensor_descriptor_type = tensor_descriptor_type
    language.load_tensor_descriptor = load_tensor_descriptor
    language.store_tensor_descriptor = store_tensor_descriptor

    language.semantic.arange = arange
    language.semantic.floordiv = floordiv
    language.semantic.atom_red_typechecking_impl = atom_red_typechecking_impl
    language.semantic.atomic_cas = atomic_cas
    language.semantic.atomic_max = atomic_max
    language.semantic.atomic_min = atomic_min
    language.semantic._load_legacy = _load_legacy
    language.semantic.maximum = maximum
    language.semantic.minimum = minimum
    language.semantic.invert = invert
    language.semantic.logical_and = logical_and
    language.semantic.logical_or = logical_or
    language.semantic.mod = mod
    language.semantic.not_ = not_
    language.semantic.and_ = and_
    language.semantic.or_ = or_
    language.semantic.xor_ = xor_
    language.semantic.minus = minus
    language.semantic.dot_scaled = dot_scaled

    language.umulhi = umulhi
    language.exp = exp
    language.exp2 = exp2
    language.log = log
    language.log2 = log2
    language.cos = cos
    language.sin = sin
    language.sqrt = sqrt
    language.sqrt_rn = sqrt_rn
    language.rsqrt = rsqrt
    language.div_rn = div_rn
    language.erf = erf
    language.tanh = tanh
    language.floor = floor
    language.ceil = ceil
    language.core.dtype.to_ir = dtype_to_ir
    language.fma = fma
    language.math.umulhi = umulhi
    language.math.exp = exp
    language.math.exp2 = exp2
    language.math.log = log
    language.math.log2 = log2
    language.math.cos = cos
    language.math.sin = sin
    language.math.sqrt = sqrt
    language.math.sqrt_rn = sqrt_rn
    language.math.rsqrt = rsqrt
    language.math.div_rn = div_rn
    language.math.erf = erf
    language.math.tanh = tanh
    language.math.floor = floor
    language.math.ceil = ceil
    language.math._check_dtype = _check_dtype
    language.math.fma = fma
    language.math.isnan = language.extra.ascend.libdevice.isnan
    language.math.isinf = language.extra.ascend.libdevice.isinf
    language.math.reciprocal = language.extra.ascend.libdevice.reciprocal
    language.math.log1p = language.extra.ascend.libdevice.log1p
    language.math.relu = language.extra.ascend.libdevice.relu
    language.math.tan = language.extra.ascend.libdevice.tan
    language.math.atan = language.extra.ascend.libdevice.atan
    language.math.tanh = language.extra.ascend.libdevice.tanh
    language.math.ilogb = language.extra.ascend.libdevice.ilogb
    language.math.ldexp = language.extra.ascend.libdevice.ldexp
    language.math.pow = language.extra.ascend.libdevice.pow
    language.math.flip = language.extra.ascend.libdevice.flip
    language.math.atan2 = language.extra.ascend.libdevice.atan2
    language.math.div_rz = language.extra.ascend.libdevice.div_rz
    language.math.fmod = language.extra.ascend.libdevice.fmod
    language.math.trunc = language.extra.ascend.libdevice.trunc
    language.math.round = language.extra.ascend.libdevice.round
    language.math.finitef = finitef
    language.math.isfinited = isfinited
    language.math.rint = rint
    language.math.atan2 = atan2
    language.extra.ascend.libdevice.umulhi = language.math.umulhi
    language.extra.ascend.libdevice.exp = language.math.exp
    language.extra.ascend.libdevice.exp2 = language.math.exp2
    language.extra.ascend.libdevice.log = language.math.log
    language.extra.ascend.libdevice.log2 = language.math.log2
    language.extra.ascend.libdevice.cos = language.math.cos
    language.extra.ascend.libdevice.sin = language.math.sin
    language.extra.ascend.libdevice.sqrt = language.math.sqrt
    language.extra.ascend.libdevice.sqrt_rn = language.math.sqrt_rn
    language.extra.ascend.libdevice.rsqrt = language.math.rsqrt
    language.extra.ascend.libdevice.div_rn = language.math.div_rn
    language.extra.ascend.libdevice.erf = language.math.erf
    language.extra.ascend.libdevice.tanh = language.math.tanh
    language.extra.ascend.libdevice.floor = language.math.floor
    language.extra.ascend.libdevice.ceil = language.math.ceil
    language.extra.ascend.libdevice.fdiv = language.math.fdiv
    language.extra.ascend.libdevice.fma = language.math.fma
    language.extra.ascend.libdevice.abs = language.math.abs