import subprocess
import os
import textwrap
import inspect
import tempfile
import csv
from io import StringIO
from pathlib import Path
import contextlib
import math
import statistics
import triton.runtime as runtime


def flagtree_do_bench(fn, warmup=25, rep=100, quantiles=None, return_mode="mean"):
    bench = FlagtreeBench(warmup=warmup, rep=rep, quantiles=quantiles, return_mode=return_mode)
    bench.do_bench(fn=fn)
    return bench._get_index()


'''
    function _quantile and _summarize_statistics is from .testing.
    if used directly, it will lead to circular dependencies
'''


def _quantile(a, q):
    n = len(a)
    a = sorted(a)

    def get_quantile(q):
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(q) for q in q]


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)


'''
    IndentedBuffer Referred to
    https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/utils/code_utils.py::IndentedBuffer
'''


class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent=0):
        self._lines = []
        self._indent = initial_indent

    def getvalue(self) -> str:
        buf = StringIO()
        for line in self._lines:
            assert isinstance(line, str)
            buf.write(line)
            buf.write("\n")
        return buf.getvalue()

    def clear(self):
        self._lines.clear()

    def __bool__(self):
        return bool(self._lines)

    def prefix(self):
        return " " * (self._indent * self.tabwidth)

    def newline(self):
        self.writeline("\n")

    def writeline(self, line):
        if line.strip():
            self._lines.append(f"{self.prefix()}{line}")
        else:
            self._lines.append("")

    def tpl(self, format_str, **kwargs):
        assert isinstance(format_str, str), "format_str must be string of type."
        format_str = format_str.format(**kwargs)
        lines = format_str.strip().splitlines()
        for line in lines:
            line = line.replace("\t", " " * self.tabwidth)
            self.writeline(line)

    def writelines(self, lines):
        for line in lines:
            self.writeline(line)

    def writemultiline(self, s):
        self.writelines(s.splitlines())

    def indent(self, offset=1):

        @contextlib.contextmanager
        def ctx():
            self._indent += offset
            try:
                yield
            finally:
                self._indent -= offset

        return ctx()


'''
    FlagtreeBench using ncu to measure performance
'''


class FlagtreeBench:

    def __init__(self, warmup=100, rep=100, quantiles=None, return_mode="mean", metrics='gpu__time_duration'):
        if FlagtreeBench.check_ncu():
            self.metrics = metrics
            self.warmup = warmup
            self.rep = rep
            self.quantiles = quantiles
            self.return_mode = return_mode
            self.function_paths = []
            self.import_modules = []
            self._get_package_path()
            self._create_temp_file()

    staticmethod

    def check_ncu():
        cmd = ["ncu", "--query-metrics"]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print("[INFO]: ncu check successfully")
            return True
        except Exception as err_msg:
            print(f"[Hint] The inability to invoke ncu on this machine"
                  f"might be due to issues such as the absence of ncu, "
                  f"lack of permissions, or a version that is too low. Specifically {err_msg}")
            return False

    @staticmethod
    def is_triton_jit_decorated(obj, max_depth=10):
        '''
            attrs temporarily adds specialized support to the kernel of flag_gems.
            About flag_gems see https://github.com/flagos-ai/FlagGems
        '''
        attrs = ['AnonymousLibTunerImpl', 'LibEntry', 'JITFunction']
        if hasattr(obj, '__class__') and obj.__class__.__name__ in attrs:
            return True

    def _get_kernels(self, _fn):
        import ast
        source = inspect.getsource(_fn)
        tree = ast.parse(source)
        globals_dict = _fn.__globals__
        calls = []

        class CallVisitor(ast.NodeVisitor):

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    calls.append({'name': node.func.id})
                elif isinstance(node.func, ast.Attribute):
                    calls.append({'name': node.func.attr})
                self.generic_visit(node)

        visitor = CallVisitor()
        visitor.visit(tree)
        jit_funcs = []
        for call in calls:
            name = call['name']
            if name not in globals_dict:
                continue
            entity = globals_dict[call['name']]
            if callable(entity):
                module = __import__(entity.__module__)
            else:
                module = entity
            _path = module.__file__
            self.function_paths.append(_path)
            module_name = _path.split('/')[-1]
            module_name = Path(module_name).stem
            self.import_modules.append((name, module_name))
            for name in dir(module):
                if name.startswith('__'):
                    continue
                obj = getattr(module, name)
                if FlagtreeBench.is_triton_jit_decorated(obj=obj):
                    jit_funcs.append(name)
        self.triton_funcs = jit_funcs

    def _get_package_path(self):
        self.user_package_path = os.environ.get('BENCH_MODULE_PATH', '')

    def _create_temp_file(self):
        self.python_exec = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
        self.python_exec.close()

        self.out_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        self.out_csv.close()

    def _write_script(self, script):
        with open(self.python_exec.name, 'w+') as f:
            f.write(script)

    def _exec(self):
        runtime.driver.active.clear_cache(self.bench_cache)
        cmd = [
            "ncu", "--metrics", self.metrics, "--csv", "--log-file", self.out_csv.name, "python3", self.python_exec.name
        ]
        print(f"[INFO]: ncu running on {self.python_exec.name}")
        subprocess.run(cmd, capture_output=True, check=True)

    def _get_index(self):
        # indexs = ['avg', 'max', 'min', 'sum']
        _index_package = {}
        kernel_name = ''
        with open(self.out_csv.name, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                for jit in self.triton_funcs:
                    if jit in row:
                        index_name = row[12].split('.')[-1]
                        index_val = float(row[14]) / 1e6
                        kernel_name = jit
                        if jit not in _index_package:
                            _index_package.update({jit: {index_name: index_val}})
                        else:
                            _index_package[jit].update({index_name: index_val})
        return _index_package[kernel_name]['avg']

    def _gen_import_and_path(self, script_code: IndentedBuffer, path_mode='insert'):
        sys_path_action_str = '0, '
        if path_mode == 'insert':
            script_code.writeline('import torch')
            script_code.writeline('import os')
            script_code.writeline('import sys')
        else:
            sys_path_action_str = ''
        if self.user_package_path != '':
            script_code.writeline(f"sys.path.{path_mode}({sys_path_action_str}'{self.user_package_path}')")
        for path in self.function_paths:
            if not os.path.isdir(path):
                path = os.path.dirname(path)
            script_code.writeline(f"sys.path.{path_mode}({sys_path_action_str}'{path}')")
        if path_mode == 'insert':
            for module_message in self.import_modules:
                fn, module = module_message
                script_code.writeline(f"from {module} import {fn}")

    def _generate_script(self, fn):
        fn_src_code_string = textwrap.dedent(inspect.getsource(fn))
        script_code = IndentedBuffer()
        self._gen_import_and_path(script_code, path_mode='insert')

        script_code.writeline(fn_src_code_string)
        script_code.writeline(f'{fn.__name__}()')
        script_code.writeline("torch.cuda.synchronize()")

        self._gen_import_and_path(script_code, path_mode='remove')
        self.script = script_code.getvalue()
        self._write_script(self.script)

    def _pre_operation(self, fn):
        '''
            Referred to triton.testing.do_bench
        '''
        di = runtime.driver.active.get_device_interface()
        fn()
        di.synchronize()
        cache = runtime.driver.active.get_empty_cache_for_benchmark()

        # Estimate the runtime of the function
        start_event = di.Event(enable_timing=True)
        end_event = di.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            runtime.driver.active.clear_cache(cache)
            fn()
        end_event.record()
        di.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5

        # compute number of warmup and repeat
        n_warmup = max(1, int(self.warmup / estimate_ms))
        # n_repeat = max(1, int(self.rep / estimate_ms))

        self.bench_cache = cache
        for _ in range(n_warmup):
            fn()

    def do_bench(self, fn):
        '''
            Measure the GPU kernel time of fn() using ncu.
            Generate a temporary Python file and then run it with 'ncu'.
        '''
        self._get_kernels(fn)
        self._generate_script(fn=fn)
        self._pre_operation(fn=fn)
        self._exec()
        self.index_set = self._get_index()
