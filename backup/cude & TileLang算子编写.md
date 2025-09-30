1. 算子命名
2. rag内容read


gpu的hopper架构指的是：
Hopper = 4N 工艺 + 800 亿晶体管 + FP8 Transformer 引擎 + 900 GB/s NVLink，把大模型训练/推理的矩阵乘、通信、内存带宽全部拉满，相对 Ampere 实现 3-9 倍 AI 提速

编写工具：

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

# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import argparse
import functools
import json
import os
import sys

import torch
import torch_npu

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..', ".."))
sys.path.append(parent_directory)

from example.common.security.path import get_valid_read_path, get_write_directory
from example.common.security.type import check_number
from example.common.utils import SafeGenerator, cmd_bool
from example.common.rot_utils.rot_qwen import rot_model
from msmodelslim.tools.copy_config_files import copy_config_files, modify_config_json
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.utils.logging import set_logger_level
from msmodelslim import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="The path of float model and tokenizer"),
    parser.add_argument('--save_path', type=str, help="The path to save quant model"),
    parser.add_argument('--layer_count', type=int, default=0, help="Layer count when loading model")
    parser.add_argument('--anti_dataset', type=str, default="./anti_prompt_50.json",
                        help="The calib data for anti outlier")
    parser.add_argument('--calib_dataset', type=str, default="./calib_prompt_50.json",
                        help="The calib data for calibration")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for anti and calibration")
    parser.add_argument('--mindie_format', action="store_true", help="Enable only mindie config save")
    parser.add_argument('--trust_remote_code', type=cmd_bool, default=False)
    parser.add_argument('--rot', action='store_true', help="rot model")
    return parser.parse_args()


def custom_hook(model_config):
    model_config["quantize"] = "w8a8_dynamic"


def get_calib_dataset_batch(model_tokenizer, calib_list, batch_size, device="npu"):
    calib_dataset = []
    calib_list = [calib_list[i:i + batch_size] for i in range(0, len(calib_list), batch_size)]
    for calib_data in calib_list:
        inputs = model_tokenizer(calib_data, return_tensors='pt', padding=True).to(device)
        calib_dataset.append(
            [value.to(device) for key, value in inputs.data.items() if isinstance(value, torch.Tensor)])
    return calib_dataset


def main():
    args = parse_args()
    set_logger_level("info")

    model_path = args.model_path
    batch_size = args.batch_size

    save_path = get_write_directory(args.save_path, write_mode=0o750)
    check_number(batch_size, int, 1, 16, "batch_size")

    safe_generator = SafeGenerator()

    config = safe_generator.get_config_from_pretrained(model_path=model_path,
                                                       trust_remote_code=args.trust_remote_code)
    num_layer = config.num_hidden_layers
    if args.layer_count < 0 or args.layer_count > num_layer:
        raise ValueError(
            f"Invalid value for parameter layer_count: {args.layer_count}."
            f"Must be between 0 and {num_layer}."
        )
    # Set layer count to 0 means use all layers, otherwise it will only use the first layer_count layers
    config.num_hidden_layers = args.layer_count if args.layer_count != 0 else config.num_hidden_layers
    # Disable use cache because we don't need to use cache, otherwise it will use too much device memory then cause OOM
    config.use_cache = False

    tokenizer = safe_generator.get_tokenizer_from_pretrained(model_path=model_path,
                                                             config=config,
                                                             trust_remote_code=args.trust_remote_code,
                                                             use_fast=True,
                                                             add_eos_token=True)

    model = safe_generator.get_model_from_pretrained(model_path=model_path,
                                                     config=config,
                                                     trust_remote_code=args.trust_remote_code,
                                                     device_map={
                                                         "model.embed_tokens": 0,
                                                         "model.layers": "cpu",
                                                         "model.norm": "cpu",
                                                         "lm_head": 0,
                                                     },
                                                     torch_dtype="auto",
                                                     attn_implementation='eager')

    anti_dataset_path = get_valid_read_path(args.anti_dataset, "json", is_dir=False)
    calib_dataset_path = get_valid_read_path(args.calib_dataset, "json", is_dir=False)
    with open(anti_dataset_path, "r") as file:
        anti_prompt = json.load(file)
    with open(calib_dataset_path, "r") as file:
        calib_prompt = json.load(file)
    anti_dataset = get_calib_dataset_batch(tokenizer, anti_prompt, batch_size, model.device)
    dataset_calib = get_calib_dataset_batch(tokenizer, calib_prompt, batch_size, model.device)

    with torch.no_grad():

        test_prompt = "what is deep learning?"
        test_input = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        if args.layer_count > 0:
            ori_out = model(**test_input)

        if args.rot:
            rot_model(model)

        if args.layer_count > 0:
            rot_out = model(**test_input)
            loss = torch.nn.MSELoss()
            logger.info(loss(ori_out[0], rot_out[0]))

    with torch.no_grad():
        anti_config = AntiOutlierConfig(w_bit=8,
                                        a_bit=8,
                                        anti_method='m4',
                                        dev_type='npu',
                                        dev_id=model.device.index)
        anti_outlier = AntiOutlier(model, calib_data=anti_dataset, cfg=anti_config)
        anti_outlier.process()

    disable_names = []
    for ids in range(config.num_hidden_layers):
        disable_names.append("model.layers." + str(ids) + ".mlp.gate")

    quant_config = QuantConfig(
        a_bit=8,
        w_bit=8,
        disable_names=disable_names,
        dev_type='npu',
        dev_id=model.device.index,
        act_method=1,
        pr=1.0,
        w_sym=True,
        mm_tensor=False,
    )

    calibrator = Calibrator(model,
                            quant_config,
                            calib_data=dataset_calib,
                            disable_level="L0",
                            mix_cfg={
                                "*.mlp.*": "w8a8_dynamic",
                                "*": "w8a8"
                            })
    calibrator.run()

    if args.mindie_format:
        quant_model_description_json_name = "quant_model_description_w8a8_dynamic.json"
    else:
        quant_model_description_json_name = "quant_model_description.json"

    save_type = "safe_tensor" if args.mindie_format else "ascendV1"
    calibrator.save(save_path,
                    json_name=quant_model_description_json_name,
                    safetensors_name="quant_model_weight_w8a8_dynamic.safetensors",
                    save_type=[save_type],
                    part_file_size=4)

    custom_hooks = {
        'config.json': functools.partial(modify_config_json, custom_hook=custom_hook)
    }
    copy_config_files(input_path=model_path, output_path=save_path, quant_config=quant_config,
                      mindie_format=args.mindie_format, custom_hooks=custom_hooks)


if __name__ == "__main__":
    # torch_npu will fork a new process to init,
    # it's lazy_init will fail after we load a big model,so we need to init it here
    torch_npu.npu.init()
    # Invoke main process
    main()
以上代码主要逻辑是啥？

</p>
</details> 