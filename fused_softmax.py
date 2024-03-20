import torch
import triton
import triton.language as tl


# x shape [M, N]
@torch.jit.script
def jit_softmax(x):

    # read MN, write M
    x_max = x.max(dim=1)[0]

    # read MN + M, write MN
    x = x - x_max[:, None]

    # read MN, write MN
    exp_x = torch.exp(x)

    # read MN, write M
    sum_exp_x = exp_x.sum(dim=1)

    # read MN + M, write MN
    softmax_x = exp_x / sum_exp_x[:, None]

    # read 5MN + 2M write 3MN + 2M
    return softmax_x


@triton.jit
def softmax_kernel(
    out_ptr,
    input_ptr,
    output_row_stride,
    input_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):

    row_id = tl.program_id(axis=0)

    col_offset = tl.arange(0, BLOCK_SIZE)

    mask = col_offset < n_cols
    row_data = tl.load(
        input_ptr + row_id * input_row_stride + col_offset,
        mask=mask,
        other=-float("inf"),
    )

    row_max = tl.max(row_data, axis=0)
    row_sub = row_data - row_max
    row_exp = tl.exp(row_sub)
    row_sum = tl.sum(row_exp, axis=0)

    out = row_exp / row_sum

    tl.store(out_ptr + row_id * output_row_stride + col_offset, out, mask=mask)


def triton_softmax(x: torch.Tensor):

    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 4096:
        num_warps = 16

    output = torch.empty_like(x)

    softmax_kernel[(n_rows,)](
        output,
        x,
        output.stride(0),
        x.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# data = torch.randn(4, 5).cuda()

# jit_res = jit_softmax(data)
# torch_res = torch.softmax(data, dim=1)
# triton_res = triton_softmax(data)
# print(torch_res - jit_res)
# print(torch_res - triton_res)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 100)
        ],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "triton",
            "torch-native",
            "torch-jit",
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch (native)",
            "Torch (jit)",
        ],  # label name for the lines
        styles=[("blue", "-"), ("green", "-"), ("green", "--")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch-native":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.softmax(x, axis=-1), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_softmax(x), quantiles=quantiles
        )
    if provider == "torch-jit":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: jit_softmax(x), quantiles=quantiles
        )
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, print_data=True)
