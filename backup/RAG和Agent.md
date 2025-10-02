# RAG 检索增强生成（Retrieval-Augmented Generation，RAG）
RAG的基本结构有哪些呢？
- 向量化模块：用来将文档片段向量化。
- 文档加载和切分模块：用来加载文档并切分成文档片段。
- 数据库：存放文档片段及其对应的向量表示。
- 检索模块：根据 Query（问题）检索相关的文档片段。
- 大模型模块：根据检索到的文档回答用户的问题。

**RAG流程**
RAG = “先把相关文档搜出来，再让大模型照着读”；流程固定为 编码→检索→拼装→生成→后处理 五步，离线再配套切块、建索引、存元数据即可上线。

RAG（检索增强生成）是通过“外部知识库检索+大模型生成”解决幻觉问题的技术，核心是让模型回答基于指定事实。  

### 我的项目拆解（技术深度体现）：  
#### 1. 多文档格式解析: 针对多种不同类型的文档包括表格、PPT、PDF 和流程说明相关的图片信息等，结合格式特性 
进行差异化解析与切分，PDF/PPT 用专用库提取文本（PyPDF2、python-pptx），表格类转换为结构化数据（如 
Pandas）或 Markdown 保留逻辑关系，图片类通过 OCR（PaddleOCR）提取和提炼成文字信息，最后按语义或固定长度分块，确保信息完整性和检索适配性。 


2. **混合检索+Rerank层精排**   
关键词（BM25+）+语义（Embedding）互补，解决专业术语OOV问题；
**两阶段排序**：
阶段一是混合检索，先用BM25+做文本召回（就是关键词匹配），使用BGE-large-1.5对长查询重点使用向量检索，对两者取并集后再去重，以确保关键字匹配和语义近似的文档都能被召回。Top5 
召回率从 0.6 提升至 0.78，MRR 从 0.52 提高至 0.63；

BM25以及BM25+区别：
BM25 只认字面 token，BM25+ 先“拼写-拼音-子词-同义词”四轮扩展，再 BM25 打分，从而把「字面不同但意思一样」的文档也捞回来。
<details><summary>Details</summary>
<p>

| 维度      | 经典 BM25    | BM25+                 |
| ------- | ---------- | --------------------- |
| 输入单元    | 精确分词 token | token + 子词 + 拼音 + 同义词 |
| OOV 专业词 | 查无此词 → 0 分 | 拼音/子词仍能命中             |
| 同义表达    | 必须关键词相同    | 同义词词典自动归一             |
| 实现成本    | 零，开箱即用     | 轻量预处理 < 5 ms          |
| 召回效果    | 易漏同义/拼写变体  | 同义/拼写/缩写全覆盖           |

BM25+ 不是改公式，而是给 BM25 前加一层“泛化 token”生成器，让它在关键词通道里就能初步解决同义和 OOV 问题，再与向量路并集互补。

```python
pip install rank-bm25 hanlp jieba pypinyin
import jieba, pypinyin, json, hanlp
from rank_bm25 import BM25Okapi
from pypinyin import lazy_pinyin
from typing import List

# 1. 同义词典（金融示例）
synonym = {
    "信用卡": ["贷记卡", "credit card"],
    "逾期": ["违约", "欠款未还"]
}

# 2. 预处理：子词 + 拼音 + 同义词
def expand(text: str) -> str:
    # 同义词替换
    for k, v in synonym.items():
        for w in v:
            text = text.replace(w, k)
    # 拼音（首字母）
    py = "".join(lazy_pinyin(text, style=pypinyin.FIRST_LETTER))
    # 子词（2-gram）
    tokens = list(jieba.cut(text))
    sub_words = ["".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    return tokens + sub_words + list(py)   # 三路融合

# 3. 建库（假设已清洗）
docs = ["信用卡逾期后利息如何计算",
        "贷记卡违约金收费标准",
        "信用卡最低还款额说明"]
corpus = [expand(d) for d in docs]
bm25 = BM25Okapi(corpus)

# 4. 查询
query = "credit card 违约利息"
query_tokens = expand(query)
scores = bm25.get_scores(query_tokens)
topk = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]
print("BM25+ Top5:", [(docs[i], s) for i, s in topk])

``` 

</p>
</details> 


embedding模型：
<details><summary>Details</summary>
<p>

**Embedding层优化**  
- **模型选型**：轻量级通用句向量模型：BERT-base（通用语义）与Sentence-BERT（句向量专项），**多语言MiniLM-L6-v2**（量化后仅80MB），在业务数据集上召回率提升12%（余弦相似度阈值0.75→0.82）。  long-context 模型：**BGE-large-1.5**、Conan-embedding-v2

- **工程优化**：通过FAISS的IVF_HNSW索引将检索延迟从500ms压至80ms，支持100万级文档库（分块策略：按标点+滑动窗口100字符，重叠20字符避免语义割裂）。  

</p>
</details> 

第二阶段是用Cross-Encoder，也就是rerank模型做语义精排，综合语义、上下文和任务特征，将最相关的文档排到前列，减少噪声干扰。效果是Top10准确率从78%→91%（对比单Embedding检索），MRR 从 0.52 提高至 0.78。  

上面的rerank模型，可以是bge-small，或者为了追求更高精度，可把 rerank 模型换成 BGE-reranker-large（+2 % MRR，-10 QPS）。

- **效率权衡**：对召回结果采样前50条进入Rerank，而非全量计算，用vllm部署INT8量化模型，单query处理耗时控制在150ms内。  

3. 多轮对话上下文管理: 为解决原有系统上下文对话能力弱的情况，设计 Context Injection 机制，对用户的跟进提 问自动关联上下文实体并替换代词，这种实体识别 + 指代消解 + 语义桥接（HanLP（NER命名实体识别。） + bert-coref-chinese（指代） + 规则桥接）的策略显著提升了多轮对话的连贯性。 

<details><summary>Details</summary>
<p>

```python
#NER
pip install hanlp
import hanlp
hanlp_ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
spans = hanlp_ner("我想查下我的信用卡账单")   # [{'text':'信用卡','type':'PROD'}]
中文 13 类实体（人名/机构/产品/金额/日期等），F1≈94%，GPU 下 5 ms。

#指代消解
pip install coref-hoi==1.0   # 已打包模型
from coref_hoi import pipeline
coref = pipeline("zh-coref-bert-base")
clusters = coref("我的信用卡逾期了，它还能用吗？")   # 返回 [[(0,3),(7,8)]]  # 信用卡-它
单句 10 ms；若置信度 < 0.6 自动回退“最近实体”规则，防止误链。

#语义桥接（替换/补全）
纯规则 + 依存检查，不跑模型：

- 用 HanLP 依存句法判断主语/宾语空缺；
- 按「实体池权重 = 出现次数 + 时间衰减」选 Top1 实体；
- 原地字符串替换或句首插入，生成新 query。

``` 
</p>
</details> 


4. **业务闭环验证**  
- **离线指标**：构建行业测试集（医疗QA 5000对），生成答案的事实一致性F1分数（反映“生成答案有多真实、多完整”）达89%（对比纯大模型提升23%）。  
- **线上优化**：针对高频问题缓存检索结果，缓存命中率达35%，整体系统QPS提升40%。  

F1计算公式：
Precision = TP / (TP + FP)   → 生成的事实里多少是对的  
Recall    = TP / (TP + FN)   → 所有该说的事实里说了多少  
F1        = 2 · P · R / (P + R)

TP = 命中事实数
FP = 生成答案里多出来的虚假/错误事实数
FN = 金标准里未被召回的事实数

提升23%原因：与纯大模型对比
纯大模型常“自由发挥”→ FP 高、遗漏关键点 → FN 也高；本项目用检索+重排+引用限制，先生成有出处的答案，再算事实 F1，因此 FP 大幅下降，F1 提升 23 个百分点。


# LLM Agent
Agent 的工作流程如下：

接收用户输入。
调用大模型（如 Qwen），并告知其可用的工具及其 Schema。
如果模型决定调用工具，Agent 会解析请求，执行相应的 Python 函数。
Agent 将工具的执行结果返回给模型。
模型根据工具结果生成最终回复。
Agent 将最终回复返回给用户。

agent如何记住历史信息：
「在线用向量/Key-Value 做短时检索 → 定期摘要/结构化转长时记忆 → 必要时把高价值数据再喂回模型权重」