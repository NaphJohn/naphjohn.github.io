1. 算子命名
2. rag内容read


gpu的hopper架构指的是：
Hopper = 4N 工艺 + 800 亿晶体管 + FP8 Transformer 引擎 + 900 GB/s NVLink，把大模型训练/推理的矩阵乘、通信、内存带宽全部拉满，相对 Ampere 实现 3-9 倍 AI 提速

编写工具：
1. TileLang = Python 前端 + TVM 编译器 + Tile 级原语，让你用 1/3 代码量写出匹配手写汇编的算子，Hopper/Ada/AMD 通用，是 2025 年最值得关注的高性能 AI Kernel DSL
https://github.com/tile-ai/tilelang