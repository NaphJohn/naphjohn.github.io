# 算子介绍
### reducescatter算子
ReduceScatter 是一个在大规模并行计算（尤其是深度学习训练）中非常重要的集合通信（Collective Communication）算子。
1. Reduce（归约）： 多个设备（如GPU）各自有一个数据块（例如一个Tensor）。归约操作（如求和sum、求最大值max、求平均值avg）将这些设备上对应位置的元素进行合并，最终得到一个全局归约后的数据块。这个结果通常只存在于一个设备上（如 Reduce 操作）或者被广播到所有设备上（如 AllReduce 操作）。
2. Scatter（散射）： 一个设备（通常是根节点）拥有一个完整的数据块，它将这个数据块切分成若干份，然后将每一份发送给一个特定的设备。
<details><summary>Details</summary>
<p>
举例说明（4个设备，操作是 sum）：

设备0 有数据 [A0, A1, A2, A3]
设备1 有数据 [B0, B1, B2, B3]
设备2 有数据 [C0, C1, C2, C3]
设备3 有数据 [D0, D1, D2, D3]

执行 ReduceScatter(sum) 后，每个设备得到的结果是：
设备0 得到：sum(A0, B0, C0, D0)
设备1 得到：sum(A1, B1, C1, D1)
设备2 得到：sum(A2, B2, C2, D2)
设备3 得到：sum(A3, B3, C3, D3)

每个设备最终都只拥有全局求和结果的一个片段。
</p>
</details> 
它的主要价值在于优化性能和内存使用，特别是在后续操作需要分散后的数据时。

### GEMM、GEMV
GEMM：矩阵×矩阵，计算密集，用 TensorCore 做分块乘加，算力瓶颈。
GEMV：矩阵×向量，内存密集，带宽瓶颈，本质是“一批列向量同时点积”。
Transformer 训练/大批推理 ≈ 100 % GEMM；单 token 小批推理里形态上是 GEMV，但实现仍走 GEMM kernel

## 算子编写工具：

| 工具           | 语言           | 硬件                  | 成熟度   | 典型场景                   | 2025 市占  |
| ------------ | ------------ | ------------------- | ----- | ---------------------- | -------- |
| **CUDA**     | C/C++        | NVIDIA only         | ★★★★★ | 手写 peak kernel、驱动库     | 70%      |
| **CUTLASS**  | C++ template | NVIDIA only         | ★★★★☆ | cuBLAS 底层、厂商库          | 60%      |
| **Triton**   | Python       | NVIDIA + AMD ROCm   | ★★★★☆ | LLM 推理/训练 fused kernel | 40%      |
| **TileLang** | Python       | NVIDIA + AMD + 沐曦\* | ★★★☆☆ | 研究原型→产品 kernel         | 5% 但增速最快 |

1. TileLang = Python 前端 + TVM 编译器 + Tile 级原语，让你用 1/3 代码量写出匹配手写汇编的算子，Hopper/Ada/AMD 通用，是 2025 年最值得关注的高性能 AI Kernel DSL
https://github.com/tile-ai/tilelang



# 量化
        ┌-------------------- 一层 Transformer --------------------┐
        |                                                        |
        |  1)  Pre-Norm                                        |
        |  2)  Attention (Q/K/V)                               |
        |  3)  Post-Attention Norm                             |
        |  4)  MoE-FFN Block                                   |
        |        ┌-----------------------------┐                |
        |        │  a)  Gate  (Linear)         │◄-- FP16 保留  |
        |        │  b)  Softmax + Top-K        │                |
        |        │  c)  Router / Load-Balance  │                |
        |        │  d)  Expert-0  (Linear×2)   │◄-- W8A8 量化  |
        |        │  e)  Expert-1  (Linear×2)   │                |
        |        │  ...                        │                |
        |        │  f)  Weighted Add (Top-K)   │                |
        |        └-----------------------------┘                |
        |                                                        |
        └--------------------------------------------------------┘

  输入校准集
      │
      ▼
[Anti-Outlier 预处理]
      │  1) 跑前向 → 记录 99.9 % 分位
      │  2) 生成 clamp_thr
      ▼
  激活 tensor ──► clamp(x, -thr, thr) ──► 计算 scale ──► 量化
      │                                              ▲
      ▼                                              │
  权重 tensor ──► per-channel scale ──► 直接 int8 ──┘
      │
      ▼
  逐层校准 (MSE)  ←-- 50 条 calib prompt
      │
      ▼
  导出混合策略：
      ├── gate 层 (Linear)          → **FP16**  （不量化）
      ├── expert-0/1/... down/up    → **W8A8 dynamic**
      └──其余投影层                 → **W8A8 static**

gate 只占 MoE 层里“一根红线”大小的 Linear，但它是决定 token 走哪条专家的投票器；保持 FP16 即可让雪崩风险归零，而内存代价可忽略不计。

<details><summary>Details</summary>
<p>

# 常见基础算子
**add_rms_norm**
算子功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。
AddRmsNorm算子将RmsNorm前的Add算子融合起来，减少搬入搬出操作。
<details><summary>Details</summary>
<p>

add_rms_norm(x, residual, weight, epsilon)
参数说明：

x(计算输入)： 数据类型支持FLOAT、FLOAT16、BFLOAT16、shape支持1-8维度，数据格式支持ND。
residual(计算输入)： 数据类型支持FLOAT、FLOAT16、BFLOAT16、shape支持1-8维度，数据格式支持ND。shape需要与x1保持一致。
weight(计算输入)： 数据类型支持FLOAT、FLOAT16、BFLOAT16、shape支持1-8维度，数据格式支持ND。shape需要与x后几维保持一致。
epsilon(计算输入)： 公式中的输入eps，用于防止除0错误。

</p>
</details> 

**Rope算子**
算子功能：推理网络为了提升性能，将query和key两路算子融合成一路。
rope(query, key, cos, sin)


**quantize算子**
y = quantize(x, scale, offset)，对应计算公式：ascend_quant(x)=round((x∗scale)+offset)
对应参数说明：
<details><summary>Details</summary>
<p>

x(输入)：需要做量化的输入。数据类型支持：FLOAT,FLOAT16,BFLOAT16(仅昇腾910B AI处理器、昇腾910C AI处理器支持)。支持ND。
scale(输入)：量化中的scale值。数据类型支持：FLOAT,FLOAT16,BFLOAT16(仅昇腾910B AI处理器、昇腾910C AI处理器支持)。scale为1维张量或多维(多维时，除最后一维，其他维度为1)，最后一维shape的大小等于输入self的最后一个维度的大小。支持ND。
offset(输入)：反向量化中的offset值。offset为1维张量(多维时，除最后一维，其他维度为1)，最后一维shape的大小等于输入x的最后一个维度的大小。数据类型支持：FLOAT,FLOAT16,BFLOAT16(仅昇腾910B AI处理器、昇腾910C AI处理器支持)，且数据类型与scale的数据类型一致。支持ND。
y(输出): int8。

</p>
</details> 

**DynamicQuant算子**
yOut, scale_out = dynamic_quant(x, smoothScales)

scaleOut=row_max(abs(x))/127  
yOut=round(x/scalOut)其中row_max代表每行求最大值，参数说明如下：
<details><summary>Details</summary>
<p>

x(计算输入): 算子输入的Tensor，shape维度要大于1，数据类型支持FLOAT16、BFLOAT16，支持ND。
smoothScalesOptional(计算输入): 算子输入的smoothScales，shape维度是x的最后一维，数据类型支持FLOAT16、BFLOAT16，支持ND。
yOut(计算输出): 量化后的输出Tensor，shape维度和x保持一致，数据类型支持INT8，暂不支持非连续的Tensor，支持ND。
scaleOut(计算输出): 量化使用的scale，shape维度为x的shape剔除最后一维，数据类型支持FLOAT，暂不支持非连续的Tensor，支持ND。

</p>
</details> 

**IFA算子**
self-attention(自注意力)利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为n的输入样本序列x，x的每个元素都是一个d维向量,可以将每个d维向量看作一个token embedding,将这样一条序列经过3个权重矩阵变换得到3个维度为n*d的矩阵。
代码调用如下
<details><summary>Details</summary>
<p>

from cann_ops import incre_flash_attetion

incre_flash_attetion(query, key_cache, value_cache,
                     None, None,
                     context_lens,
                     None, None, None, None, None, None, None,
                     block_tables, None,
                     num_heads,
                     scale,
                     kv_heads)

</p>
</details> 

linear算子
out = linear(x1, x2, bias)
