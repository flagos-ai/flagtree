import os
import shutil
from pathlib import Path
from setup_tools.utils.tools import flagtree_root_dir, flagtree_submodule_dir, DownloadManager, Module

def precompile_hook_flir(*args, **kargs):
    default_backends = kargs["default_backends"]
    if 'amd' in default_backends:
        default_backends.remove('amd')
    default_backends.append('flir')

downloader = DownloadManager()

submodules = (Module(name="ascendnpu-ir", url="https://gitcode.com/qq_42979146/AscendNPU-IR.git",
                     branch="ascend_with_flir",
                     dst_path=os.path.join(flagtree_submodule_dir, "ascendnpu-ir")), )


def is_compile_ascend_npu_ir():
    return os.getenv("ASCEND_NPU_IR_COMPILE", "1") == "1"
