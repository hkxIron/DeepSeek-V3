import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, List

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm


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
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


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
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:[batch, seq_len]
        if world_size > 1:
            # 1. 小于vocab_start_idx或大于等于vocab_end_idx的token，即它们不在本worker的token，其embedding为0
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            # 2. 将token ID转换为本地索引（减去起始偏移量）
            x = x - self.vocab_start_idx
            # 3. 不属于本GPU的token ID置零（避免越界）
            x[mask] = 0

        # 4. 本地embedding查找（无效token会取到self.weight[0]）
        # y:[batch, seq_len, dim]
        y = F.embedding(x, self.weight)
        if world_size > 1:
            # 5. 将无效token的embedding置零
            y[mask] = 0
            # 6. 跨GPU求和（所有GPU的y相加得到完整结果）
            # dist.all_reduce：通过NCCL通信汇总所有GPU的局部结果（求和）。
            # y:[batch, seq_len, dim], 即y中某些token处的embedding为0,需要从其它GPU的y中取得, 求取即可获取
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight.element_size() > 1:
        # y= x@w.T + bias
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        # 反量化
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        # 整数量化
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
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
    # NOTE:在输出维度上并行，因此称为列并行
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0
        # 只使用部分输出维度，即out_features // world_size
        self.part_out_features = out_features // world_size
        super().__init__(in_features, out_features=self.part_out_features, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:[batch, seq_len, in_features]
        # y:[batch, seq_len, in_features/world_size]
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    # NOTE:在输入维度上并行，因此称为行并行
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0
        # 只使用部分输入维度，即out_features // world_size
        self.part_in_features = in_features // world_size
        super().__init__(in_features=self.part_in_features, out_features=out_features, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:[batch, seq_len, in_features]
        # y:[batch, seq_len, in_features/world_size]
        y = linear(x, self.weight)
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

    def forward(self, x: torch.Tensor):
        # x: [batch_size, seq_len, hidden_dim]
        x = x.float()
        # rsqrt: 用于计算输入张量元素的平方根的倒数, 等价于1/torch.sqrt(x)，但 torch.rsqrt 通常经过优化，计算效率更高（尤其在GPU上）。
        normed_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # 只除了标准差
        return normed_x.type_as(self.weight) * self.weight

"""
注意：这里也用到了yarn rope
"""
def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # freqs: base**(2*i/10000)
    # inv_freq shape: [dim / 2], 其值为: 1 / [ 10000 ^ (even_dim_index / dim) ], 只取偶数维度
    inv_freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        inv_freqs = inv_freqs / factor * (1 - smooth) + inv_freqs * smooth

    # t: [0, 1, 2, ,,,,, seq_len-1], shape[seq_len]
    t = torch.arange(seqlen)
    # inv_freqs: [seq_len, dim//2]
    position_dim_theta = torch.outer(t, inv_freqs)
    # 生成旋转复向量，[seq_len, dim//2], 不同位置不同维度的旋转复向量不同
    freqs_cis = torch.polar(abs=torch.ones_like(inv_freqs), angle=position_dim_theta)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    # x: [bsz, seqlen, n_local_heads, qk_head_dim]
    # =>view: [bsz, seqlen, n_local_heads, qk_head_dim//2, 2], 将最后一个维度拆成[head_dim//2, 2]， 最后一维的2个维度分别对应复数实部和虚部
    # =>view_as_complex: [bsz, seqlen, n_local_heads, qk_head_dim//2, 2]
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    # freqs_cis.view: [bsz=1, seq_len, n_local_heads=1, dim//2], 旋转复向量
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    # rotated_x: 对x向量进行旋转，在复数域内相乘后，又恢复成实数域
    # [bsz, seqlen, n_local_heads, qk_head_dim//2, 2]
    # =>view_as_real: [bsz, seqlen, n_local_heads, qk_head_dim//2, 2]
    # => flatten(3):  [bsz, seqlen, n_local_heads, qk_head_dim]
    rotated_x = torch.view_as_real(x * freqs_cis).flatten(3)
    return rotated_x.to(dtype)


"""
注意：MLA就昌借鉴了lora先降维再升维的思想， 所以里面的变量名都是lora_XX
"""
class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim #  无位置信息的向量
        self.qk_rope_head_dim = args.qk_rope_head_dim # 有位置信息的向量
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            # 不对q进行先降维再升维(低秩压缩)
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            # 对q进行先降维再升维(低秩压缩)
            # x * wq_a * wq_b, 就是MLA, 即先降维到latent空间，再升维
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(dim=self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        # wkv_a降维
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        # wkv_b升维
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))

        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            """
            yarn中：
            sqrt(1/t) = 0.1*ln(s) + 1.0, 它们是分别乘在query与key上的
            """
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            # 乘以在最后 softmax上，需要平方
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        # x: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_size = x.size()
        # 在prefill阶段，可能seq_len>1
        # 但在decode阶段，seq_len=1
        end_pos = start_pos + seq_len
        if self.q_lora_rank == 0: # 对q不使用MLA
            q = self.wq(x)
        else:
            # 对q使用MLA, 先降维再升维
            # q: [batch_size, seq_len, hidden_dim]
            q = self.wq_b(self.q_norm(self.wq_a(x)))

        q = q.view(batch_size, seq_len, self.n_local_heads, self.qk_head_dim)

        # q: [batch_size, seq_len, n_local_heads, qk_head_dim = qk_nope_head_dim + qk_rope_head_dim]
        # q_nope: 无位置信息的向量, [batch_size, seq_len, n_local_heads, qk_nope_head_dim]
        # q_rope: 有位置信息的向量, [batch_size, seq_len, n_local_heads, qk_rope_head_dim]
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rope = apply_rotary_emb(q_rope, freqs_cis)

        # kv: [batch_size, seq_len, dim= = kv_lora_rank + qk_rope_head_dim]
        kv = self.wkv_a(x)
        # kv: [batch_size, seq_len, kv_lora_rank]
        # kv_rope: [batch_size, seq_len, qk_rope_head_dim]
        kv, k_rope = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # kv_rope: [batch_size, seq_len, head_num=1, qk_rope_head_dim]
        k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freqs_cis)

        if attn_impl == "naive":
            # q_nope: 无位置信息的向量, [batch_size, seq_len, n_local_heads, qk_nope_head_dim]
            # q_rope: 有位置信息的向量, [batch_size, seq_len, n_local_heads, qk_rope_head_dim]
            # q: [batch_size, seq_len, n_local_heads, qk_head_dim = qk_nope_head_dim + qk_rope_head_dim]
            q = torch.cat([q_nope, q_rope], dim=-1)
            # kv: [batch_size, seq_len, kv_lora_rank]
            # => kv: [batch_size, seq_len, n_local_heads * (qk_nope_head_dim + v_head_dim)]
            kv = self.wkv_b(self.kv_norm(kv))
            # kv: [batch_size, seq_len, n_local_heads, (qk_nope_head_dim + v_head_dim)]
            kv = kv.view(batch_size, seq_len, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            # kv: [batch_size, seq_len, n_local_heads, (qk_nope_head_dim + v_head_dim)]
            # => k_nope: [batch_size, seq_len, n_local_heads, qk_nope_head_dim]
            # => v: [batch_size, seq_len, n_local_heads, v_head_dim]
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # k: [batch_size, seq_len, n_local_heads, qk_head_dim = qk_nope_head_dim + qk_rope_head_dim]
            k = torch.cat([k_nope, k_rope.expand(-1, -1, self.n_local_heads, -1)], dim=-1)

            self.k_cache[:batch_size, start_pos:end_pos] = k # 更新kv cache缓存
            self.v_cache[:batch_size, start_pos:end_pos] = v
            # 即[s,d] @ [t, d].T => [s, t]
            # q:[batch_size, query_seq_len, head_num, head_dim]
            # k:[batch_size, key_seq_len, head_num, head_dim]
            # scores:[batch_size, query_seq_len, head_num, key_seq_len]
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:batch_size, :end_pos]) * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            self.kv_cache[:batch_size, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:batch_size, start_pos:end_pos] = k_rope.squeeze(2)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:batch_size, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_rope, self.pe_cache[:batch_size, :end_pos])) * self.softmax_scale
        # 计算attention中的casual mask
        if mask is not None:
            scores += mask.unsqueeze(1) # mask掉的地方均为-inf

        # scores:[batch_size, key_seq_len, head_num, key_seq_len]
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            # 即[s,t] @ [t, d] => [s, d]
            # scores:[batch_size, query_seq_len, head_num, key_seq_len]
            # v: [batch_size, seq_len, n_local_heads, v_head_dim]
            #  => x: [batch_size, query_seq_len, head_num=n_local_heads, v_head_dim]
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:batch_size, :end_pos])
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:batch_size, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])

        # x: [batch_size, query_seq_len, head_num=n_local_heads, v_head_dim]
        # flatten => [batch_size, query_seq_len, head_num * v_head_dim]
        # => [batch_size, query_seq_len, dim]
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.up_proj = ColumnParallelLinear(dim, inter_dim)
        self.gate_proj = ColumnParallelLinear(dim, inter_dim)
        self.down_proj = RowParallelLinear(inter_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 这个与llama的做法一些区别
        """
        llama: y = down( silu(gate(x)) * up(x) )

        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        return self.down_proj(F.silu(self.up_proj(x)) * self.gate_proj(x))


class ExpertGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts # 激活6个专家
        self.n_groups = args.n_expert_groups # 1
        self.topk_groups = args.n_limited_groups # 1
        self.score_func = args.score_func
        self.route_scale = args.route_scale # 1
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # y= x@w.T + bias
        # x:[batch_size*seq_len, dim]
        # scores:[batch_size*seq_len, n_routed_experts]
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            # scores:[batch_size*seq_len, n_routed_experts]
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()

        # original_scores:[batch_size*seq_len, n_routed_experts]
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias

        # 若有多个group
        if self.n_groups > 1:
            # scores:[batch_size*seq_len, n_routed_experts]
            # => scores:[batch_size*seq_len, n_groups, n_routed_experts//n_groups]
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                """
                amax等价于 torch.max(scores, dim=-1).values
                amax 在 CUDA 设备上会调用优化后的内核
                """
                # group_scores:[batch_size*seq_len, n_groups, 1]
                group_scores = scores.amax(dim=-1)
            else:
                # group_scores:[batch_size,*seq_len,n_groups, 1]
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            """
            目标：从 scores 张量中提取每组（group）的前 topk_groups 个最高得分，并通过掩码（mask）将这些位置保留，其余位置置零。
            输入：
            scores：形状为 [batch_size*seq_len, n_groups, n_routed_experts/n_groups] 的张量，表示每个组的得分。
            group_scores：每组得分的汇总（如最大值或平均值）。
            输出：处理后的 scores，仅保留每组前 topk_groups 个最高得分的位置。
            将 src（这里是 True）按 index（indices）填充到 mask 的指定位置。
            
            scores:[batch_size*seq_len, n_groups, n_routed_experts/n_groups]
            mask:[batch_size*seq_len, n_groups]
            """
            mask = torch.zeros_like(scores[..., 0]).scatter_(dim=1, index=indices, src=True)
            # scores:[batch_size*seq_len, n_groups, n_routed_experts/n_groups]
            # => [batch_size*seq_len, n_routed_experts]
            scores = (scores * mask.unsqueeze(-1)).flatten(1)

        # scores:[batch_size*seq_len, n_routed_experts=64]
        # => indices [batch_size*seq_len, topk=6]
        indices = torch.topk(scores, self.topk, dim=-1)[1]

        # 从dim=1上用indices去收集scores
        # weights:[batch_size*seq_len, topk=6]
        weights = original_scores.gather(dim=1, index=indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)

        # weights:[batch_size*seq_len, topk=6]
        weights *= self.route_scale
        return weights.type_as(x), indices


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
        self.n_routed_experts = args.n_routed_experts # 路由的专家有61个
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.expert_gate = ExpertGate(args)
        self.experts:List[Expert] = nn.ModuleList([Expert(args.dim, args.moe_intermediate_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        shape = x.size()
        # x: [batch*seq_len, dim]
        x = x.view(-1, self.dim)
        # weights:[batch_size*seq_len, topk=6]
        # indices:[batch_size*seq_len, topk=6]
        weights, indices = self.expert_gate.forward(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        # NOTE：专家路由时，使用的是for,而不是固定的矩阵相乘后再mask
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert.forward(x[idx]) * weights[idx, top, None]

        # 还有一个共享专家，是一定需要计算的
        # x: [batch*seq_len, dim]
        # z: [batch*seq_len, n_shared_experts*moe_intermediate_dim]
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)


class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.intermediate_size) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # rms_norm + casual_attention + resudial
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        # rms_norm + mlp + resudial
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()

        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))

        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None

        if seqlen > 1:
            # 生成3下三角矩阵
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(diagonal=1)

        for layer in self.layers:
            h = layer.forward(h, start_pos, freqs_cis, mask)

        h = self.norm(h)[:, -1]
        logits = self.head(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
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
