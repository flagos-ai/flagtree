import torch
import triton
import triton.language as tl
import triton.experimental.flagtree.language as fl
import triton.experimental.flagtree.edsl.cuda.language as flcuda

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@fl.dialect(name="cuda")
def edsl(x, y, output):
    tidx = flcuda.get_thread_id(0)


@fl.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.zeros_like(x)
    fl.call(edsl, [x, y, output])
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    x = torch.randn(1024, device=DEVICE)
    y = torch.randn(1024, device=DEVICE)
    z = add(x, y)
