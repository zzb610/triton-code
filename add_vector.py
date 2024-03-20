import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(out_ptr, a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0)  # 1D grid
    data_start = pid * BLOCK_SIZE
    data_offset = data_start + tl.arange(0, BLOCK_SIZE)
    mask = data_offset < n_elements

    a_data = tl.load(a_ptr + data_offset, mask=mask)
    b_data = tl.load(b_ptr + data_offset, mask=mask)

    out = a_data + b_data

    tl.store(out_ptr + data_offset, out, mask=mask)


def add(a, b):

    out = torch.empty_like(a)

    n_elements = out.numel()

    BLOCK_SIZE = 64

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    add_kernel[grid](out, a, b, n_elements=n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(12, 28, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="vector-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y), quantiles=quantiles
        )
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True)

# size = 98432

# a = torch.rand(size=(size,), device="cuda")
# b = torch.rand(size=(size,), device="cuda")

# out = add(a, b)
# torch_out = torch.add(a, b)

# print(torch.allclose(out, torch_out))
# print(out - torch_out)
