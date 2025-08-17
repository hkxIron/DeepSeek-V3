from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config

"""
实现了一个 Triton 内核（GPU 加速的量化操作），用于对输入张量 x 进行动态范围量化（Dynamic Range Quantization），并将量化后的结果存储到 y，同时记录缩放因子 s。以下是逐行解析：
量化公式：y =x/s, 其中 s = max(|x|)/448, 是一种对称的最大值量化

目的：将输入 x 动态缩放到一个固定范围（类似 int8 量化，但保留浮点精度）。
特点：
分块量化：每个线程块独立计算自己的缩放因子 s。
动态范围：缩放因子 s 取决于输入数据的绝对值最大值（动态适应数据分布）

这种量化方法常用于：
模型推理加速：将激活值（activations）动态量化为低精度（如 int8），减少计算和内存开销。
数据预处理：在训练前对输入数据进行归一化。
通过 Triton 实现，可以高效利用 GPU 并行计算能力，显著提升量化速度。

缩放因子448:
    448 是一个经验值（可能是为了将值映射到 int8 的对称范围 [-127, 127]，
    由 4 位指数（Exponent） 和 3 位尾数（Mantissa） 组成（另有 1 位符号位）。其名称中的 fn 表示 "no infinity"（不包含无穷大），是 NVIDIA 针对 GPU 计算优化的一种格式。以下是其关键特性：
    格式分解
    位分配	1（符号）	4（指数）	3（尾数）
    作用	sign	exponent	mantissa
    - 指数偏移（Bias）：7（与 IEEE float16/float32 不同，后者为 15/127）。
    - 尾数隐含的整数位：默认 1（正规数），但部分情况可能为 0（非正规数）。

    最大正规数（Normal Maximum）：
    指数：1110（二进制，即 14 - 7 = 7）。
    尾数：111（二进制，即 1.111，十进制 1.875）。 
    值：+/- 1.875*2^7 = +/-240

    最小正规数（Normal Minimum）
    指数：0001（二进制，即 1 - 7 = -6）。
    尾数：000（二进制，即 1.0）。
    值：+/-1.0*2^(-6) = +/- 0.015625

    非正规数（Subnormal）：
    指数：0000，尾数非零。
    值： +/- 0.mantissa * 2^(-6)
    最小非零值: 1.001 = +/- 0.125*2^(-6) = +/- 0.001953125

    零（Zero）：
    指数和尾数均为 0，表示 +0.0 或 -0.0。

    特殊值：
    1111 指数保留用于 NaN（无 Infinity）

    4. 为什么代码中使用 448 作为缩放基准？
    在之前的量化代码中，缩放因子计算为：
    s = max(|x|)/448

    448 的由来：
    float8_e4m3fn 的理论最大值是 240，但实际使用中可能因硬件优化或数值稳定性需要，选择更大的安全系数（例如 448 ≈ 240 × 1.87）。 

    确保量化后的值 y = x / s 不会溢出 float8 范围, 即 |y| <=448/s =  240

    实际应用注意事项
    1.精度损失：
        float8_e4m3fn 的尾数仅 3 位，相对 float16/float32 会丢失大量精度。
        适合对噪声不敏感的场景（如神经网络激活值）。
    2.硬件支持：
        需 NVIDIA Hopper 架构（如 H100 GPU）或 AMD MI300 等支持 float8 的硬件。
    3.动态量化：
        分块量化（如代码中的 block_size）可减少极端值的影响，但需存储额外的缩放因子 s
"""
@triton.jit # @triton.jit：表示这是一个 Triton 的即时编译（JIT）内核，会在 GPU 上运行。
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    参数：
        x_ptr：输入张量 x 的指针（待量化的数据）。
        y_ptr：输出张量 y 的指针（量化后的数据）。
        s_ptr：缩放因子 s 的指针（每个块的最大绝对值除以 448）。
        BLOCK_SIZE：常量，表示每个 GPU 线程块处理的元素数量。
    """

    """
    pid：当前线程块的全局索引（沿 axis=0 的维度）。
    offs：当前线程块处理的输入数据偏移量（offsets），计算公式为：
        offs=pid×BLOCK_SIZE+[0,1,…,BLOCK_SIZE−1]
    例如，若 pid=2 且 BLOCK_SIZE=128，则 offs 的范围是 [256, 257, ..., 383]。
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    """
    tl.load(x_ptr + offs)：从全局内存加载输入数据 x 的当前块, 注意：只加载当前块的数据
       .to(tl.float32)：将数据转换为 float32 类型（确保计算精度）。
    s：缩放因子，计算公式为：s = max(|x|)/448
    在块内取最大值，然后进行缩放 
    """
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448. # 最大值量化

    """
    y = x / s：将输入数据按缩放因子 s 归一化（近似映射到 [-448, 448] 范围）。
    y.to(y_ptr.dtype.element_ty)：将结果转换为输出张量y的数据类型（例如 int8）
    """
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)

    """
    tl.store(y_ptr + offs, y)：将量化后的数据 y 写回全局内存。
    tl.store(s_ptr + pid, s)：存储当前块的缩放因子 s（每个线程块对应一个 s 值）
    """
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


"""
它使用 Triton 内核 (act_quant_kernel) 对输入张量 x 进行分块量化，并返回量化后的张量 y 和每块的缩放因子s
"""
def activation_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入：
    x：输入张量（需为连续内存且最后一维可被 block_size 整除）。
    block_size：分块大小（默认 128）。

    输出：
    y：量化后的张量（float8_e4m3fn 格式）。
    s：每块的缩放因子（float32 格式）。
    """

    """
    x.is_contiguous()：确保输入张量内存连续（Triton 要求）。
    x.size(-1) % block_size == 0：确保最后一维可被 block_size 整除（避免边界处理）。
    """
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0

    """
    s: 缩放因子， (*x.shape[:-1], num_blocks)，其中 num_blocks = x.size(-1) // block_size
    """
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    """
    cdiv(x,y）:即向上取整整除， 即ceil(x/y), 算出num_blocks
    act_quant_kernel：Triton JIT 编译的内核（见前文解析）。
    [grid_fn]：指定网格大小（线程块数量）。
        参数传递：
        x：输入张量。
        y：输出量化张量。
        s：缩放因子张量。
        BLOCK_SIZE：分块大小。
    
    """
    grid_fn = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    act_quant_kernel[grid_fn](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    实现了一个 权重反量化（Weight Dequantization） 的 Triton 内核，用于将量化后的权重张量 x 结合缩放因子 s 还原为原始精度的浮点数值。以下是详细解析：

    x_ptr：量化后权重的指针（通常是 int8 或 float8）。
    s_ptr：缩放因子指针（float32）。
    y_ptr：反量化后输出的指针（float32）。
    M, N：输入张量 x 的行数和列数。
    BLOCK_SIZE：线程块处理的块大小。
    """

    """
    每个线程块处理一个 BLOCK_SIZE x BLOCK_SIZE 的子矩阵。
    """
    pid_m = tl.program_id(axis=0) # 行方向的线程块ID
    pid_n = tl.program_id(axis=1) # 列方向的线程块ID
    n = tl.cdiv(N, BLOCK_SIZE) # 列方向的块数量
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) # 行偏移
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) # 列偏移
    offs = offs_m[:, None] * N + offs_n[None, :] #全局偏移, 二维展开成一维向量
    # 边界掩码 要求行下标不越界，且列下标不越界
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32) # 加载量化权重
    # 缩放因子 s 的存储布局假设为每块一个值（s.shape = (M//BLOCK_SIZE, N//BLOCK_SIZE)
    s = tl.load(s_ptr + pid_m * n + pid_n) # 加载对应的缩放因子
    y = x * s   # 反量化
    tl.store(y_ptr + offs, y, mask=mask)  # 存储结果


"""
反量化

量化：
x_quant = round(x_float/s)

反量化：
y = x_quant*s ~= x_float
"""
def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    # x: [row, col]
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    """
    将大矩阵分割为许多小矩阵
    网格尺寸为 (M//BLOCK_SIZE, N//BLOCK_SIZE) ,  动态计算的
    """
    grid_fn = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid_fn](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


"""
这段代码实现了一个 FP8 GEMM（General Matrix Multiply） 的 Triton 内核，支持自动调优（autotune）和动态缩放因子处理。以下是详细解析：

参数组合：
BLOCK_SIZE_M：行方向的分块大小（16/32/64）。
BLOCK_SIZE_N：列方向的分块大小（32/64/128）。
BLOCK_SIZE_K：固定为 128（K 维度分块）。
num_stages：流水线阶段数（3-6），影响寄存器使用和延迟隐藏。
num_warps：固定为 8（每个线程块的 warp 数量）。
作用：Triton 会根据输入尺寸（N, K）自动选择最优配置。
"""
fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m,
            'BLOCK_SIZE_N': block_n,
            'BLOCK_SIZE_K': 128},
           num_stages=num_stages,
           num_warps=8)
    for block_m in [16, 32, 64]
        for block_n in [32, 64, 128]
            for num_stages in [3, 4, 5, 6]
]

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    #jit动态编译时会赋常量值
                    M: tl.constexpr,
                    N: tl.constexpr,
                    K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    计算矩阵乘法 C=A⋅B，其中：
    A 和 B 是 FP8 格式的输入矩阵。 通过缩放因子 a_s 和 b_s 动态反量化。

    a_ptr, b_ptr：FP8 输入矩阵的指针。
    c_ptr：输出矩阵指针（通常为 float32）。
    a_s_ptr, b_s_ptr：缩放因子指针（每块一个值）。
    M, N, K：矩阵尺寸, a:[M,K], B:[K, N]
    BLOCK_SIZE_*：分块大小（由 autotune 选择）。
    """
    pid_m = tl.program_id(axis=0) # 行方向的线程块ID, 每个线程块计算一个 BLOCK_SIZE_M x BLOCK_SIZE_N 的子矩阵。
    pid_n = tl.program_id(axis=1) # 列方向的线程块ID
    # 在内积维度划分为k个block
    k_num = tl.cdiv(K, BLOCK_SIZE_K) # k个block

    """
    offs_m 和 offs_n 通过取模处理边界条件。
    offs_m:[BLOCK_SIZE_M]
    offs_n:[BLOCK_SIZE_N]
    offs_k:[BLOCK_SIZE_K]
    """
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 取出所遥a的元素指针
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :] # A 的指针
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None] # B 的指针
    a_s_ptrs = a_s_ptr + offs_m * k_num # A 的缩放因子指针
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k_num # B 的缩放因子指针

    # NOTE:acc用的是float32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # 每次只加截向量的一小块进行计算
    for i in range(k_num):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        """
        子矩阵乘法, 先在fp8内计算乘法，再反量化为float16
        C = (A* a_s) * (B * b_s).T
        """
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask) # 回写到c


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    # a:[M, K], b:[K, N], c:[M, N]
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid_fn = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp8_gemm_kernel[grid_fn](a, b, c, a_s, b_s, M, N, K)
    return c

if __name__ == "__main__":
    x = torch.randn(1024, device='cuda')  # 输入数据
    y, s = activation_quant(x, block_size=128)  # 量化
    print(y)
    print(s)
    #x_dequant = y.float() * s.repeat_interleave(128, dim=-1)[:x.numel()]

    # 反量化
    # 假设 x 是量化后的权重（如 int8），s 是分块缩放因子
    x = torch.randint(-128, 127, (1024, 1024), device='cuda', dtype=torch.int8)
    s = torch.randn((1024//128, 1024//128), device='cuda', dtype=torch.float32)

    # 反量化
    y = weight_dequant(x, s)  # y 是 float32 张量
    print(y.shape)  # 输出: torch.Size([1024, 1024])
