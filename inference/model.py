import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, List

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import activation_quant, weight_dequant, fp8_gemm

"""
代码非常值得一读，大神写的代码就是不一样
"""

world_size = 1 # GPU总数（全局变量）
rank = 0 # 当前GPU的编号（0到world_size-1）
block_size = 128 #
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    #vocab_size: int = 102400 # 原始
    vocab_size: int = 102400//100 # 本地debug

    dim: int = 2048
    intermediate_size: int = 10944
    moe_intermediate_dim: int = 1408
    #n_layers: int = 27 # 原始
    n_layers: int = 3 # 本地debug

    n_dense_layers: int = 1
    n_heads: int = 16

    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.

    # mla, NOTE：只是用lora的思想来实现MLA
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn RoPE
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32 # 旋转得快的周期
    beta_slow: int = 1 # 旋转得慢的周期
    mscale: float = 1. # 不知道是什调节因子，一般为1


"""
目标：将完整的词表（vocab_size）均匀拆分到多个GPU上，每个GPU只存储一部分词向量，减少单卡显存占用。
"""
class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0
        # 每个进程上的vocab_size
        self.part_vocab_size = (vocab_size // world_size)
        # 每个进程上的vocab_start_idx
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        # 注意: embedding只申请一部分的weight
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:[batch, seq_len]
        if world_size > 1:
            # 1. 小于vocab_start_idx或大于等于vocab_end_idx的token，即它们不在本worker的token，其token_id直接设为0
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            # 2. 将token ID转换为本地索引（减去起始偏移量）
            x = x - self.vocab_start_idx
            # 3. 不属于本GPU的token ID置零（避免越界）
            x[mask] = 0

        # 4. 本地embedding查找（无效token会取到self.weight[0]）
        # x:[batch, seq_len]
        # y:[batch, seq_len, dim]
        y = F.embedding(x, self.weight)
        if world_size > 1:
            # 5. 将mask部分的无效token的embedding置零
            y[mask] = 0
            # 6. 跨GPU求和（所有GPU的y相加得到完整结果）
            # dist.all_reduce：通过NCCL通信汇总所有GPU的局部结果（求和）。
            # y:[batch, seq_len, dim], 即y中某些token处的embedding为0,需要从其它GPU的y中取得, 求取即可获取
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight.element_size() > 1:
        """
        element_size是 PyTorch 中的一个方法，用于返回张量（Tensor）中 单个元素所占的字节数（bytes per element）。它可以帮助你计算张量的内存占用情况
        torch.float32 (float)	4
        torch.int16 (short)	2
        torch.int8	1
        """
        # y= x@w.T + bias
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        # 权重反量化
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        # 激活值fp8_e4m3量化
        x, scale = activation_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    dtype = torch.bfloat16 # 默认为bf16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 若传入dtype=None时，dtype=bf16
        # weight:[out_feature, in_feat]
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        """
        element_size是 PyTorch 中的一个方法，用于返回张量（Tensor）中 单个元素所占的字节数（bytes per element）。它可以帮助你计算张量的内存占用情况
        torch.float32 (float)	4
        torch.int16 (short)	2
        torch.int8	1
        """
        if self.weight.element_size() == 1: # 8bit量化
            # weight只有一个字节，说明其为量化后的权重
            scale_out_features = (out_features + block_size - 1) // block_size # 向量取整整除，ceil(x/block_size)
            scale_in_features = (in_features + block_size - 1) // block_size
            # 在weight上强行加一个scale
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.part_out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    # NOTE:在wegiht输出维度上并行，因此称为列并行
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0
        # NOTE:可以看到，列并行只是在输出维度上将向量分为多个块
        # 只使用部分输出维度，即out_features // world_size
        # weight:[in_features, out_feat/world_size]
        self.part_out_features = out_features // world_size
        super().__init__(in_features, out_features=self.part_out_features, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:[batch, seq_len, in_features]
        # weight:[out_feat/world_size, in_feature]
        # => y = x @ w.T
        # y:[batch, seq_len, in_features/world_size]
        y = linear(x, self.weight, self.bias)
        return y

class RowParallelLinear(Linear):
    # NOTE:在weight输入维度上并行，因此称为行并行
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0
        # 只使用部分输入维度，即out_features // world_size
        self.part_in_features = in_features // world_size
        # weight: [out_features, part_in_features]
        super().__init__(in_features=self.part_in_features, out_features=out_features, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE:y是先计算了其中的一部分y,然后all_reduce_sum得到所有worker的y
        # x:[batch, seq_len, in_features//world_size]
        # weight: [out_features, part_in_features]
        # y = x @ w.T
        # y:[batch, seq_len, out_features]
        y = linear(x, self.weight)

        # NOTE:因为将内积向量拆开了，需要再求和得到真正的向量内积
        #      行并行需要求和
        if world_size > 1:
            dist.all_reduce(y, op=dist.ReduceOp.SUM)

        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    """
    rmsnorm: y = x / std(x) * alpha, 注意，没有减均值
    """
    def forward(self, x: torch.Tensor):
        # x: [batch_size, seq_len, hidden_dim]
        x = x.float()
        # rsqrt: 用于计算输入张量元素的平方根的倒数, 等价于1/torch.sqrt(x)，但 torch.rsqrt 通常经过优化，计算效率更高（尤其在GPU上）。
        # std(x) = sqrt(sum(x^2)/n)
        normed_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # 只除了标准差
        return normed_x.type_as(self.weight) * self.weight

"""
注意：这里也用到了yarn rope
计算不同位置不同维度旋转向量 [theta_d(seq_len, dim//2)
"""
def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast_rotation = args.beta_fast # 32, dim越大, 旋转角度越小，旋转得越慢, 低频
    beta_slow_ratation = args.beta_slow # 1
    base = args.rope_theta
    scale_factor = args.rope_factor

    """
    find_correction_dim 是 Yarn RoPE 中用于 动态计算需要调整频率的维度范围 的关键函数，其目的是确定在长序列外推时，哪些维度（dim）的频率需要被衰减（即应用 factor 缩放）
    
    参数说明
    num_rotations：目标旋转周期数（由 beta_fast 或 beta_slow_ratation 控制）。
    dim：RoPE 的旋转维度（通常是注意力头维度的一半，d_head // 2）。
    base：RoPE 的原始频率基（如 10000）。
    max_seq_len：预训练时的最大序列长度。
    返回值
    一个标量值，表示 需要调整频率的临界维度索引（即从该维度开始，频率需要被缩放）。
    
    2. 数学原理
    RoPE 对 Query 和 Key 的旋转操作可以展开为
    e^(i*m*theta) = cos(m*theta_i) + i * sin(m*theta_i)
    这是一个复数旋转，旋转速度为theta_i, 每增加一个token位置,旋转角度theta_i弧度
   
    theta_i = 1/base^(2i/dim) 
    
    现定义波长lambda_i为旋转2*pi弧度所需的位置跨度：
    m* theta_i = 2*pi =>
    波长= lambda_i = 2*pi/theta_i
    =>
    theta_i = 2*pi/lambda_i
    则theta_i的特理意义是空间角频率，即单位位置变化的弧度数
    
    
    (1) 原始RoPE的频率计算
    对于位置m 和维度i，复数旋转角度为： theta_i(m)= m*theta_i
    原始RoPE的频率为: theta_i = 1/base^(2i*/dim)
    
    对于维度i,旋转一个周期2*pi需要的token为：2*pi/theta_i，
    那么最大序列长度max_seq_len内其对应的旋转周期个数为： 
    num_rotation = num_rotation_i = 
    = max_seq_len/(2*pi/theta_i)
    = max_seq_len*theta_i/2*pi 
    = max_seq_len /(2*pi* base^(2i/d) )
   
    示例: 
    若base=10000, max_seq_len=4096, dim=512
    i=0, 
        theta_i = 1, 
        lambda_i = 2*pi=6.28,  
        num_rotation_i = 4096/(2*pi) = 652个周期, 称之为高频，旋转得快
    i=dim//2, 
        theta_i = 1/base=1/10000, 
        lambda_i=2*pi*10000=62800
        num_rotation_i = 4906/(2*pi*10000) = 0.0652个周期, 称之为低频，旋转得慢
    
    (2) 求解临界维度i
    对于维度i，最大序列长度max_seq_len内其对应的旋转周期个数为： 
    num_rotation = max_seq_len /(2*pi* base^(2i/d) )
    =>
    i = d/2 * log(max_seq_len/(2*pi*num_rotation)) / log(base)
    = d * log(max_seq_len/(2*pi*num_rotation)) / (2*log(base))
    这就是 find_correction_dim 的公式。
    """
    # NOTE: 对于维度i，最大序列长度max_seq_len内其对应的旋转周期个数为：num_rotations
    def find_correction_dim_by_rotation_num(num_rotations:int, dim:int, base:float, max_seq_len:int):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(high_rotation_num:int,# dim越小, 旋转角度越大，旋转得越快, 高频
                              low_rotation_num:int, # dim越大, 旋转角度越小，旋转得越慢, 低频
                              dim:int,
                              base:float,
                              max_seq_len:int):
        # dim越小, 旋转角度越大，旋转得越快, 高频, 10
        dim_of_high_rotation_num = math.floor(find_correction_dim_by_rotation_num(high_rotation_num, dim, base, max_seq_len))
        # dim越大, 旋转角度越小，旋转得越慢, 低频, 23
        dim_of_low_rotation_num = math.ceil(find_correction_dim_by_rotation_num(low_rotation_num, dim, base, max_seq_len))
        return max(dim_of_high_rotation_num, 0), min(dim_of_low_rotation_num, dim - 1)

    def linear_ramp_factor(low_dim:int, high_dim:int, dim:int):
        """
         NOTE: 生成非递减阶梯函数
            0<x<low_dim: 0
            low_dim<x<high_dim: (x-low_dim)/(high_dim-low_dim)
            high_dim<x: 1
        """
        if low_dim == high_dim:
            high_dim += 0.001
        # min=10, max=32,
        # 在10~32之间，线性增加
        linear_func = (torch.arange(dim, dtype=torch.float32) - low_dim) / (high_dim - low_dim)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # freqs: 1/base^(2i/dim)
    # freq shape: [dim / 2], 其值为: 1 / [ 10000 ^ (even_dim_index / dim) ], 只取偶数维度,
    # NOTE: 但freqs在tranformer的llama源码中叫inv_freqs, 但叫法并不准确，应访是频率
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # NOTE:使用yarn rope
    if seqlen > args.original_seq_len:
        low_dim_of_fast_rotation, high_dim_of_low_rotation = find_correction_range(beta_fast_rotation, beta_slow_ratation, dim, base=base, max_seq_len=args.original_seq_len)
        """
        低频dim维度（i > 临界值32）：旋转周期数较少（rotations_i < num_rotations），保留原始频率
        高频维度（i < 临界值=10）：旋转周期数较多（rotations_i >= num_rotations），需要衰减频率（除以 factor）。
            
        paper中公式：
        max_seq_len中包含的完整的周期数：
        r(d) = L/lambda_i

        gamma(d) ，代表外推程度
        if r(d)>beta: # beta =32, 即dim<10
            gamma(d) = 1, #beta=32, 高频外推, 不内插,就是保持原rope不变
        elif r(d)<beta and r(d)>=alpha: # 中间的部分外推, 部分内插, 10<dim<23
            gamma(d) = (r(d)-alpha)/(beta-alpha)
        else: # r(d) < alpha = 1, 即dim>23
            gamma(d) = 0  # 仰频的不外推, 完全内插
        """

        # NOTE:计算外推程度 extrapolate_degree, 1-extrapolate_degree代表内插程度
        extrapolate_factor = 1 - linear_ramp_factor(low_dim_of_fast_rotation, high_dim_of_low_rotation, dim // 2)
        # NOTE: 高频外推（啥也不干）， 低频内插（max_seq_len里一个周期还没用满）, 中间的部分内插
        #   内插是需要将原始的旋转角度缩小 scale_factor倍的
        #   而外推则啥也不干
        interpolation_freqs = freqs / scale_factor #  内推旋转角度
        freqs = interpolation_freqs * (1 - extrapolate_factor) + freqs * extrapolate_factor

    # position_idx: [0, 1, 2, ,,,,, seq_len-1], shape[seq_len]
    position_idx = torch.arange(seqlen)
    # theta_d = freqs: [seq_len, dim//2]
    position_dim_theta = torch.outer(position_idx, freqs)

    """
    NOTE: 生成旋转复向量: e^(iθ) ，[seq_len, dim//2], 不同位置不同维度的旋转复向量不同, theta_d
    freqs_cis = torch.polar(abs=torch.ones_like(freqs), angle=position_dim_theta), 其命名意思为复数旋转角度
    freqs_cis: [seq_len, dim//2]
    
    cis 是复数中表示 幅角为 θ 的单位复数 的简写形式，定义为：
    cis(θ) = e^(iθ) = cos(θ ) + i*sin(θ)
    即 c:cos, i: i, s: sin
    """
    freqs_cis = torch.polar(abs=torch.ones_like(position_dim_theta), angle=position_dim_theta)
    return freqs_cis


"""
对x进行旋转复向量freqs_cis角度操作
freqs_cis: [seq_len, dim//2]

1. 先将x转为复数，然后乘以旊转复向量e^(i*theta)
2. 然后将复数恢复为实数
"""
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    # x: [bsz, seq_len, n_local_heads, qk_head_dim]
    # =>view: [bsz, seq_len, n_local_heads, qk_head_dim//2, 2], 将最后一个维度拆成[head_dim//2, 2]， 最后一维的2个维度分别对应复数实部和虚部
    # =>view_as_complex: [bsz, seq_len, n_local_heads, qk_head_dim//2, 2]
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    # freqs_cis: [seq_len, dim//2]
    # => freqs_cis.view: [bsz=1, seq_len, n_local_heads=1, dim//2], 旋转复向量
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))

    # rotated_x: 对x向量进行旋转角度freq_cis，在复数域内相乘后，又恢复成实数域
    # [bsz, seq_len, n_local_heads, qk_head_dim//2, 2]
    # =>view_as_real: [bsz, seq_len, n_local_heads, qk_head_dim//2, 2]
    # => flatten(3):  [bsz, seq_len, n_local_heads, qk_head_dim]
    rotated_x = torch.view_as_real(x * freqs_cis).flatten(3)
    return rotated_x.to(dtype)

"""
注意：MLA就昌借鉴了lora先降维再升维的思想， 所以里面的变量名都是lora_XX
"""
class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim # 2048维
        self.n_heads = args.n_heads # 16
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank #
        self.kv_lora_rank = args.kv_lora_rank # kv low rank, 512
        self.qk_nope_head_dim = args.qk_nope_head_dim # 无位置信息的向量, 128, 维度较高
        self.qk_rope_head_dim = args.qk_rope_head_dim # 有位置信息的向量, 64, 维度较低
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim # qk_head_dim = 128+64 = 192维
        self.v_head_dim = args.v_head_dim # 128

        if self.q_lora_rank == 0:
            # NOTE:不对q进行先降维再升维(低秩压缩), 列并行
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            # 对q进行先降维再升维(低秩压缩)
            # x * wq_a * wq_b, 就是MLA, 即先降维到latent空间，再升维
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(dim=self.q_lora_rank)
            # NOTE: 列并行
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        # NOTE: wkv_a降维
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        # NOTE: wkv_b升维, 列并行
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))

        # NOTE:行并行
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        # yarn放大系数计算
        if args.max_seq_len > args.original_seq_len:
            """
            yarn中：
            sqrt(1/t) = 0.1*ln(s) + 1.0, 它们是分别乘在query与key上的
            s = L_new/L_old
            """
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            # NOTE: 由于原始yarn放大系数是乘在query与key上的，所以乘以在最后 softmax时，需要平方
            self.softmax_scale = self.softmax_scale * mscale**2

        # NOTE: 每层layer有自己的MLA, 每层都是固定内存分配的, 而不是动态分配的
        if attn_impl == "naive":
            # k cache: [batch_size, max_seq_len, n_local_heads, qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192]
            # v cache: [batch_size, max_seq_len, n_local_heads, v_head_dim =128]
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else: # absorb, 吸收矩阵 W^(UQ).T * W(UK)
            # kv cache
            # kv_cache: [batch_size, max_seq_len, kv_lora_rank=512]
            # pe_cache: [batch_size, max_seq_len, qk_rope_head_dim=64]
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        # x: [batch_size, query_seq_len, hidden_dim]
        batch_size, seq_len, hidden_size = x.size()
        # 在prefill阶段，可能seq_len>1
        # 但在decode阶段，seq_len=1
        end_pos = start_pos + seq_len
        if self.q_lora_rank == 0: # 对q不先降维再升维(低秩压缩)
            # x: [batch_size, query_seq_len, hidden_dim]
            # wq:[dim, n_heads*qk_head_dim]
            # => NOTE: 通过wq的列权重并行后, q的维度是n_local_head*qk_head_dim, 而不是n_head*qk_head_dim, 需要细细体会
            # q: [batch_size, query_seq_len, n_local_heads*qk_head_dim]
            q = self.wq(x)
        else:
            # 对q使用MLA, 先降维再升维
            # NOTE: wq_b为列并行
            # q: [batch_size, query_seq_len, n_local_heads*qk_head_dim]
            q = self.wq_b(self.q_norm(self.wq_a(x)))

        # NOTE:当前rank上只有q的部分列,即 q: [batch_size, query_seq_len, n_local_head*qk_head_dim]
        # q: [batch_size, query_seq_len, n_local_heads, qk_head_dim]
        q = q.view(batch_size, seq_len, self.n_local_heads, self.qk_head_dim)

        # q: [batch_size, query_seq_len, n_local_heads, qk_head_dim = qk_nope_head_dim + qk_rope_head_dim]
        # q_nope: 无位置信息的向量, [batch_size, query_seq_len, n_local_heads, qk_nope_head_dim=128]
        # q_rope: 有位置信息的向量, [batch_size, query_seq_len, n_local_heads, qk_rope_head_dim=64]
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # q_rope: 有位置信息的向量, [batch_size, query_seq_len, n_local_heads, qk_rope_head_dim]
        q_rope = apply_rotary_emb(q_rope, freqs_cis)

        # kv: [batch_size, seq_len, dim= = kv_lora_rank + qk_rope_head_dim]
        kv = self.wkv_a(x)
        # kv: [batch_size, seq_len, kv_lora_rank=512]
        # kv_rope: [batch_size, seq_len, qk_rope_head_dim=64]
        kv, k_rope = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # k_rope: [batch_size, seq_len, head_num=1, qk_rope_head_dim=64]
        k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freqs_cis)

        if attn_impl == "naive":
            # q_nope: 无位置信息的向量, [batch_size, seq_len, n_local_heads, qk_nope_head_dim=128]
            # q_rope: 有位置信息的向量, [batch_size, seq_len, n_local_heads, qk_rope_head_dim=64]
            # q: [batch_size, seq_len, n_local_heads, qk_head_dim = qk_nope_head_dim + qk_rope_head_dim=192]
            q = torch.cat([q_nope, q_rope], dim=-1)
            # kv: [batch_size, seq_len, kv_lora_rank=512]
            # NOTE: wkv_b升维, 列并行, 输出维度为: n_local_heads * (qk_nope_head_dim + v_head_dim), 而不是n_heads
            # => kv: [batch_size, seq_len, n_local_heads * (qk_nope_head_dim + v_head_dim)]
            kv = self.wkv_b(self.kv_norm(kv))
            # kv: [batch_size, seq_len, n_local_heads, (qk_nope_head_dim + v_head_dim)]
            kv = kv.view(batch_size, seq_len, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            # kv: [batch_size, seq_len, n_local_heads, (qk_nope_head_dim + v_head_dim)]
            # => k_nope: [batch_size, seq_len, n_local_heads, qk_nope_head_dim]
            # => v: [batch_size, seq_len, n_local_heads, v_head_dim=128]
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # k_rope: [batch_size, seq_len, head_num=1, qk_rope_head_dim]
            # k_rope.expand() => [batch_size, seq_len, n_local_heads, qk_rope_head_dim]

            # k: [batch_size, seq_len, n_local_heads, qk_head_dim = qk_nope_head_dim + qk_rope_head_dim]
            k = torch.cat([k_nope, k_rope.expand(-1, -1, self.n_local_heads, -1)], dim=-1) # NOTE:这里k_rope进行广播，然后存储，浪费了内存

            # NOTE: 这里只计算了start:end之间的kv cache, 而不是所有完整序列的kv cache，在decode阶段，每次输入的seq_len=1,即只有一个token
            # k_cache: [batch_size, Key_seq_len, n_local_heads=16, qk_head_dim = qk_nope_head_dim + qk_rope_head_dim=192]
            # v_cache: [batch_size, key_seq_len, n_local_heads, v_head_dim=128]
            self.k_cache[:batch_size, start_pos:end_pos] = k # 更新kv cache缓存, 只更新当前位置的kv
            self.v_cache[:batch_size, start_pos:end_pos] = v

            # NOTE:计算的时候，是取出所有的key/value
            # 即[s,d] @ [t, d].T => [s, t]
            # q:[batch_size, query_seq_len, head_num, qk_head_dim=qk_nope_head_dim + qk_rope_head_dim=192]
            # k_cached:[batch_size, key_seq_len, head_num, qk_head_dim=qk_nope_head_dim + qk_rope_head_dim=192] , 注意：此处是取出前面所有的kv
            # => q@k =
            # scores:[batch_size, query_seq_len, head_num, key_seq_len]
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:batch_size, :end_pos]) * self.softmax_scale # 除以/sqrt(dim)
        else:
            # NOTE:默认, absorb, 将 W(UQ)_i.T@ W(UK)_i吸收在同一个低维矩阵q_nope中
            # wkv_b: [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank=512]
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
            # wkv_b: [n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank=512]
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)

            # q_nope: 无位置信息的向量, [batch_size, seq_len, n_local_heads, qk_nope_head_dim=128]
            # wkv_b[:, :self.qk_nope_head_dim]: [h_heads, qk_nope_head_dim, kv_lora_rank=512]
            # => 最后两个维度上矩阵相乘
            # q_nope:[batch_size, seq_len, n_local_heads, kv_lora_rank=512]
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])

            # kv: [batch_size, key_seq_len, kv_lora_rank=512]
            # kv_cache:[batch_size, key_seq_len, kv_lora_rank=512]
            self.kv_cache[:batch_size, start_pos:end_pos] = self.kv_norm(kv)

            # pe_cache: [batch_size, key_seq_len, qk_rope_head_dim=64]
            self.pe_cache[:batch_size, start_pos:end_pos] = k_rope.squeeze(2)

            # q_nope:[batch_size, query_seq_len, n_local_heads, kv_lora_rank=512]
            # kv_cache:[batch_size, key_seq_len, kv_lora_rank=512]
            # nope_score:[batch_size, query_seq_len, n_local_heads, key_seq_len]
            nope_score = torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:batch_size, :end_pos])

            # q_rope: [batch_size, query_seq_len, n_local_heads, qk_rope_head_dim=64]
            # pe_cache: [batch_size, key_seq_len, qk_rope_head_dim]
            # => rope_score:[batch_size, query_seq_len, n_local_heads, key_seq_len]
            rope_score =torch.einsum("bshr,btr->bsht", q_rope, self.pe_cache[:batch_size, :end_pos])

            # 将带位置信息的向量和无位置信息的向量拼接在一起
            # scores:[batch_size, query_seq_len, n_local_heads, key_seq_len]
            scores = (nope_score + rope_score) * self.softmax_scale

        # 计算attention中的casual mask
        if mask is not None:
            scores += mask.unsqueeze(1) # mask掉的地方均为-inf

        # scores:[batch_size, key_seq_len, head_num=n_local_heads, key_seq_len]
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            # 即[s,t] @ [t, d] => [s, d]
            # scores:[batch_size, query_seq_len, head_num=n_local_heads, key_seq_len]
            # v: [batch_size, key_seq_len, head_num=n_local_heads, v_head_dim]
            # => x: [batch_size, query_seq_len, head_num=n_local_heads, v_head_dim]
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:batch_size, :end_pos])
        else:
            # NOTE: absorb

            # 计算attention value
            # scores:[batch_size, query_seq_len, n_local_heads, key_seq_len]
            # kv_cache:[batch_size, key_seq_len, kv_lora_rank=512]
            # => x: [batch_size, query_seq_len, head_num=n_local_heads, kv_local_rank=512]
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:batch_size, :end_pos])

            # 对attention value再降维
            # x: [batch_size, query_seq_len, head_num=n_local_heads, kv_local_rank=512]
            # wkv_b: [n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank=512], 对应于原paper中的W^(DKV)
            # wkv_b[:,-self.v_head_dim:]: [n_heads, v_head_dim, kv_lora_rank=512]
            # => x: [batch_size, query_seq_len, head_num=n_local_heads, v_head_dim=128]
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])

        # x: [batch_size, query_seq_len, head_num=n_local_heads, v_head_dim=128]
        # x.flatten => [batch_size, query_seq_len, head_num * v_head_dim], 将所有head的拼接在一起,然后再进行线性变换
        # wo.T: [head_num * v_head_dim, dim=2048]
        # NOTE:wo行并行, 需要将多个rank的矩阵结果相加即可得到原始值
        # => [batch_size, query_seq_len, dim=2048]
        x = self.wo(x.flatten(start_dim=2, end_dim=-1))
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.up_proj = ColumnParallelLinear(dim, intermediate_dim)
        self.down_proj = RowParallelLinear(intermediate_dim, dim)
        self.gate_proj = ColumnParallelLinear(dim, intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 这个与llama的做法一些区别
        """
        NOTE: up_proj列并行, gate_proj:列并行 -> 行并行 down_proj
            1. up_proj,gate_proj列Column并行，可将输出y在最后一个维度dim上分为n个小向量
            2. down_proj行Row并行， 可将输入的y在最后一个维度的n个小向量相加all_reduce_sum后，恢复成原始向量大小，
            两者配合， 输出维度不变，对上层调用完全透明

        llama: y = down( silu(gate(x)) * up(x) )
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        silu(x) = x* sigmoid(x), 可以改善relu(x)中的梯度消失问题
        or
        silu(x) = x* 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        """
        return self.down_proj(F.silu(self.up_proj(x)) * self.gate_proj(x))


class ExpertGateWeight(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts # 激活6个专家
        self.n_groups = args.n_expert_groups # 1, 专家分为多少组
        self.topk_groups = args.n_limited_groups # 1
        self.score_func = args.score_func # 默认softmax
        self.route_scale = args.route_scale # 1
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # y= x@w.T + bias
        # x:[batch_size*seq_len, dim]
        # scores:[batch_size*seq_len, n_routed_experts=64]
        scores = linear(x, self.weight)
        if self.score_func == "softmax": # 默认 softmax
            # scores:[batch_size*seq_len, n_routed_experts=64]
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            # 但我实测过，sigmoid函数的topk效果并没有更好
            scores = scores.sigmoid()

        # original_scores:[batch_size*seq_len, n_routed_experts=64]
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias

        # 若有多个group
        if self.n_groups > 1:
            # scores:[batch_size*seq_len, n_routed_experts=64]
            # => scores:[batch_size*seq_len, n_groups, n_routed_experts//n_groups]
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                """
                amax等价于 torch.max(scores, dim=-1).values, 只返回最大值，不返回index
                amax 在 CUDA 设备上会调用优化后的内核
                """
                # scores:[batch_size*seq_len, n_groups, n_routed_experts//n_groups]
                # =>
                # group_scores:[batch_size*seq_len, n_groups]
                group_scores = scores.amax(dim=-1) # amax会降维
            else:
                # group_scores:[batch_size*seq_len, n_groups]
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1) # sum会降维

            # topk_inds:[batch_size*seq_len, topk_groups]
            topk_inds = group_scores.topk(self.topk_groups, dim=-1)[1]

            """
            tensor.scatter_(dim, index, src):
            要求： src.shape == index.shape.
                self[index[i][j][k]] [j][k] = src[i][j][k]  # if dim == 0
                self[i] [index[i][j][k]] [k] = src[i][j][k]  # if dim == 1
                self[i][j] [index[i][j][k]] = src[i][j][k]  # if dim == 2
            
            相反的操作：一般用在从input中选topk元素
            torch.gather(input, dim, index)
            要求： src.shape == index.shape.
                out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
                out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
                out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

            scores:[batch_size*seq_len, n_groups, n_routed_experts/n_groups]
                => scores[...,0]是只取第0个group的score, [batch_size*seq_len, n_groups]
            topk_inds:[batch_size*seq_len, topk_groups]
            
            =>
            mask:[batch_size*seq_len, n_groups]
            """
            mask = torch.zeros_like(scores[..., 0]).scatter_(dim=1, index=topk_inds, src=True)
            # scores:[batch_size*seq_len, n_groups, n_routed_experts/n_groups]
            # => [batch_size*seq_len, n_routed_experts=64]
            scores = (scores * mask.unsqueeze(-1)).flatten(1) # mask=0处均被设为0

        # scores:[batch_size*seq_len, n_routed_experts=64]
        # => topk_inds [batch_size*seq_len, topk=6]
        topk_inds = torch.topk(scores, self.topk, dim=-1)[1]

        # 从dim=1上用indices去收集topk scores作为weights
        # original_scores:[batch_size*seq_len, n_routed_experts=64]
        # weights:[batch_size*seq_len, topk=6]
        weights = original_scores.gather(dim=1, index=topk_inds)
        if self.score_func == "sigmoid":
            # weights:[batch_size*seq_len, 6]
            weights /= weights.sum(dim=-1, keepdim=True) # 如果是sigmoid, 则需要归一化为1, 才是真正的概率

        # weights:[batch_size*seq_len, topk=6]
        # topk_inds [batch_size*seq_len, topk=6]
        weights *= self.route_scale
        return weights.type_as(x), topk_inds


class Expert(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.up_proj = Linear(dim, intermediate_dim)
        self.gate_proj = Linear(dim, intermediate_dim)
        self.down_proj = Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.up_proj(x)) * self.gate_proj(x))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0
        self.n_routed_experts = args.n_routed_experts # 路由的专家有64个
        self.n_local_experts = args.n_routed_experts // world_size # 假设world_size=4, 则n_local_experts = 16
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts # 每个进程只处理部分专家
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.expert_weight = ExpertGateWeight(args)
        # NOTE:专家并行, 不同的专家放在不同的rank上
        self.experts:List[Expert] = nn.ModuleList([Expert(args.dim, args.moe_intermediate_dim) \
                                                       if self.experts_start_idx <= i < self.experts_end_idx else None \
                                                            for i in range(self.n_routed_experts)])
        # NOTE: 共享专家, 将多个专家的参数合并到一个MLP的中间层中了, 但并享专家内部是张量并行, 即MLP内部先列并行再行并行
        self.shared_experts = MLP(dim=args.dim, intermediate_dim=args.n_shared_experts * args.moe_intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        shape = x.size()
        # x: [batch*seq_len, dim]
        x = x.view(-1, self.dim)
        """
        先对gate进行前向传播，从gate中选出每个token的topK的专家，然后只对各个token的topK的专家进行前向传播
        """
        # expert_weights:[batch_size*seq_len, topk=6]
        # indices:[batch_size*seq_len, topk=6]
        expert_weights, indices = self.expert_weight.forward(x)
        # routed_experts_score: [batch*seq_len, dim]
        routed_experts_score = torch.zeros_like(x)

        """
        # 假设有 4 个专家（n_routed_experts = 4）
        self_n_routed_experts = 4
        # indices 表示每个 token 被路由到哪个专家, 每次选2个专家
        indices = torch.tensor([[0, 1], 
                                [1, 3], 
                                [0, 2]])
        # 统计每个专家被选中的次数
        counts = torch.bincount(indices.flatten(), minlength=self_n_routed_experts).tolist()
        print(counts)  # 输出: [2, 2, 1, 1]
        """
        # 分桶统计每个专家bin的计数
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()

        # NOTE：1. 专家路由时，使用的是for,而不是固定的矩阵相乘后再mask, 这样可以节省计算量
        #       2. 此处使用了专家并行,只处理当前world_index上的专家, 不同的专家放在不同的rank上
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0: # 若该专家没有被选中，则跳过
                continue
            expert = self.experts[i]
            """
            对于二维张量，torch.where()它会返回两个一维张量：
                sample_idx:第一个：满足条件的行索引（比如 batch 中的第几个样本）
                select_expert_idx:第二个：满足条件的列索引（比如在 select_expert_idx-k 中的第几个位置）
                
            indices = torch.tensor([
                [0, 1],
                [1, 2],
                [0, 2], # 注意：第2个样本没有选择专家1
                [1, 1]
            ])  # 形状: [4, 2]

            i = 1
            sample_idx, select_expert_idx = torch.where(indices == i)

            print(sample_idx)  # tensor([0, 1, 3, 3])   -> 第0、1、3、3个样本
            print(select_expert_idx)  # tensor([1, 0, 0, 1])   -> 在这些样本中，是第1、0、0、1个位置
            所以专家 1 被样本 [0,1,3,3] 选中，分别是在它们的 select_expert_idx-k 路由中的第 [1,0,0,1] 位。
            """
            # indices:[batch_size*seq_len, topk=6], 每个样本选择哪个专家
            # sample_idx:[batch_size*seq_len]
            # select_expert_idx:[batch_size*seq_len]
            sample_idx, select_expert_idx = torch.where(indices == i)

            # x: [batch*seq_len, dim]
            # x[sample_idx]: [token_idx, dim]
            # expert_score: [token_idx, dim]
            expert_score = expert.forward(x[sample_idx])

            # NOTE："+=", 这里是各专家对各样本token的加权打分的累加
            # expert_score: [token_idx, dim]
            # expert_weights[token_idx, select_expert_idx].shape = [token_idx]
            #  => expert_weights[token_idx, select_expert_idx, None] => [token_idx, 1]
            routed_experts_score[sample_idx] += expert_score * expert_weights[sample_idx, select_expert_idx, None]

        if world_size > 1: # NOTE: 专家并行，将各个路由专家对token的打分sum在一起
            dist.all_reduce(routed_experts_score, op=dist.ReduceOp.SUM)

        # NOTE:还有一个共享专家，是每次都需要计算的, 共享专家内部是张量并行, 所以不会在各rank中重复
        # x: [batch*seq_len, dim]
        # shared_experts_score: [batch*seq_len, dim]
        shared_experts_score = self.shared_experts.forward(x)
        # routed_experts_score: [batch*seq_len, dim]
        # shared_experts_score: [batch*seq_len, dim]
        # => return [batch, seq_len, dim]
        # NOTE: 共享专家+路由专家的权重
        return (routed_experts_score + shared_experts_score).view(shape)


class Layer(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        # NOTE: 只有前面n_dense_layers个layer使用MLP, 后面的layer使用MoE
        #   MLP: 张量并行 = 列并行 + 行并行
        #   MOE: 路由专家并行 + 共享专家张量并行
        self.ffn = MLP(args.dim, args.intermediate_size) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        NOTE：这里并没有dropout
        """
        # pre_rms_norm + casual_attention + residual
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        # pre_rms_norm + mlp + residual
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        # 重新计算world_size
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()

        self.max_seq_len = args.max_seq_len
        # NOTE: embedding行并行 - 张量并行
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers:List[Layer] = torch.nn.ModuleList()

        for layer_id in range(args.n_layers):
            self.layers.append(Layer(layer_id, args))

        self.norm = RMSNorm(args.dim)
        # NOTE: lm_head列并行 - 张量并行
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        # freq_cis = theta_d: [max_seq_len, dim//2]
        # => freq_cis[start_pos:start_pos+seqlen]: [seqlen, dim//2]
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None

        if seqlen > 1:
            # 生成3下三角矩阵
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(diagonal=1)

        for layer in self.layers:
            h = layer.forward(h, start_pos, freqs_cis, mask)

        h = self.norm(h)[:, -1]
        # NOTE: lm_head为列并行
        logits = self.head(h) # logits: [batch, vocab_size//world_size]
        if world_size > 1:
            all_logit_list = [torch.empty_like(logits) for _ in range(world_size)]
            # 将各个rank计算的logits收集至all_logit_list中
            dist.all_gather(all_logit_list, logits)
            logits = torch.cat(all_logit_list, dim=-1)
        return logits


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    #torch.set_default_device("cuda")
    torch.set_default_device("cpu")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, size=(2, 128))
    model = Transformer(args)
    print(model(x).size())
