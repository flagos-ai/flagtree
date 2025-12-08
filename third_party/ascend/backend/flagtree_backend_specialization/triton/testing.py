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
