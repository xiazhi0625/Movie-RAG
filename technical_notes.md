# Movie RAG 技术问题与解决记录

## 一、文档说明

本文记录 Movie RAG 项目开发过程中遇到的三个典型技术问题。

通过分析问题现象、定位根本原因、修改代码和验证结果，
总结项目开发过程中的工程实践经验。

---

## Bug 1：Gradio 多轮对话历史记录格式不兼容

### 1. 问题现象

项目最初可以正常完成单轮电影推荐。

但是，当用户进行第二轮追问时，例如：

第一轮：推荐一部科幻电影。

第二轮：还有其他的吗？

系统无法正常回答，终端出现以下错误：

```text
too many values to unpack (expected 2)
```

### 2. 原因分析

最初使用的历史记录处理代码为：

```python
for user_msg, ai_msg in history[-2:]:
    history_text += f"用户: {user_msg}\nAI: {ai_msg}\n"
```

这段代码假设历史记录由用户消息和 AI 消息组成的二元组构成。

但实际使用的 Gradio 版本传入的是字典列表，
其中包含 role、content 等字段。

部分 content 还可能是结构化文本块，而不是普通字符串。

因此，直接使用两个变量解包历史记录会产生错误。

### 3. 解决方法

首先检查 Gradio 实际传入的数据：

```python
print("历史记录：", repr(history), flush=True)
```

确认数据结构之后，增加 normalize_history() 函数，
将不同格式的历史消息统一转换为：

```python
[
    {"role": "user", "content": "推荐科幻电影"},
    {"role": "assistant", "content": "推荐某部电影"}
]
```

随后修改历史记录遍历方式：

```python
history_text = ""

for item in history[-6:]:
    speaker = "用户" if item["role"] == "user" else "AI"
    history_text += f"{speaker}: {item['content']}\n"
```

修改后，查询重写和最终回答生成都可以读取标准化的历史记录。

### 4. 验证结果

修复后进行了连续对话测试：

- 用户输入电影推荐请求，系统正常返回结果。
- 用户追问“还有吗？”，系统能够继续回答。
- 用户要求“再推荐一部”，系统能够生成后续推荐。

测试日志显示历史消息已经正常传递，
此前的解包错误不再出现。

但严格不重复推荐仍需要进一步验证。

### 5. 经验总结

处理第三方框架传递的数据时，不能依赖未经验证的数据结构假设。

应该先通过日志检查实际数据格式，再进行统一的数据转换。

这次修复也让我理解了多轮对话中历史记录传递、
查询重写和 Prompt 构建之间的关系。


---

## Bug 2：WSL2 环境下 Hugging Face 联网重试导致启动缓慢

### 1. 问题现象

项目运行在 Windows WSL2 Ubuntu 环境中。

启动 Gradio 程序时，终端反复出现：

```text
Network is unreachable
Connection reset by peer
Retrying in 1s [Retry 1/5]
Retrying in 2s [Retry 2/5]
Retrying in 4s [Retry 3/5]
Retrying in 8s [Retry 4/5]
Retrying in 8s [Retry 5/5]
```

即使本地已经缓存 SentenceTransformer 模型，程序仍然尝试访问 Hugging Face。

多次重试增加了启动等待时间。

重试结束后，终端能够显示：

```text
Loading weights: 100%
```

说明模型可以使用本地缓存加载。

### 2. 原因分析

项目使用的 Embedding 模型为：

`paraphrase-multilingual-MiniLM-L12-v2`

在模型初始化过程中，相关依赖可能尝试访问 Hugging Face，
检查模型配置、版本或 Adapter 文件。

但当前 WSL2 网络环境无法稳定访问 Hugging Face，
导致连接失败并触发重试。

早期代码在 retriever.py 中设置离线变量：

```python
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

然而 app.py 会先导入 Gradio 等依赖，再导入 retriever.py。

如果 Hugging Face 相关模块已提前导入，
部分环境变量可能无法按预期生效。

需要注意，环境变量的设置时机是重要嫌疑，
但没有通过独立实验完全证明它是唯一原因。

### 3. 解决方法

采用在 Python 进程启动之前设置离线环境变量的方法。

在 WSL2 终端执行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u app.py
```

这样，Python 进程及其导入的相关依赖可以在启动时读取离线配置。

同时，确保本地已经完整下载 Embedding 模型。

如果需要明确要求只从本地加载，可以在支持该参数的
SentenceTransformers 版本中使用：

```python
model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2",
    local_files_only=True
)
```

注意：离线模式不会自动下载缺失的模型文件。

### 4. 验证结果

后续测试中，项目成功完成以下流程：

- 加载本地 Embedding 模型。
- 初始化 MovieRetriever。
- 初始化 DeepSeek API 客户端。
- 启动 Gradio Web 服务。
- 完成连续多轮电影推荐。

说明项目已经能够在当前 WSL2 环境中正常运行。

但尚未进行严格的启动耗时对照测试，
因此不报告具体的性能提升百分比。

### 5. 经验总结

这次问题让我认识到：

1. 模型权重加载成功，不代表此前没有发生网络请求。
2. 环境变量的设置时机可能影响第三方库的行为。
3. 本地模型缓存和 Hugging Face 离线模式是两个相关但不同的概念。
4. 排查启动故障时，应通过分阶段日志确定程序具体执行到哪里。
5. 不应把所有启动失败都直接判断为 Gradio 死锁。


---

## Bug 3：电影语义检索与评分需求不匹配

### 1. 问题现象

项目早期采用纯语义向量检索，根据用户问题与电影文本之间的向量距离返回候选结果。

在测试中发现，当用户输入：

“评分很高的经典科幻片”

系统曾推荐《Project Almanac》等电影，其中《Project Almanac》的评分为 3.75。

虽然电影在类型上具有相关性，但并不完全符合用户对高评分的要求。

这说明单纯按照语义相似度排序，无法充分兼顾电影质量和用户的评分偏好。

### 2. 原因分析

早期检索方法主要依赖 SentenceTransformer 和 ChromaDB。

基本流程为：

1. 将用户问题转换为向量。
2. 使用 ChromaDB 搜索相似电影。
3. 根据向量距离返回 Top-K 结果。

这种方法主要考虑文本之间的语义相关性。

但是，“高评分”属于数值评价要求，向量相似度并不能保证满足这一条件。

例如：

- 电影 A：语义高度相关，但评分较低。
- 电影 B：语义较为相关，但评分较高。

如果只考虑语义相似度，电影 A 可能排在电影 B 前面。

### 3. 解决方法

第 9 周开始尝试引入语义相似度与电影评分的融合排序。

最终版本将这一方法整合进 retriever.py 的 hybrid_search() 函数。

第一步：扩大初始召回数量。

```python
results = self.collection.query(
    query_embeddings=[query_vec],
    n_results=50
)
```

先获取最多 50 部候选电影，而不是直接返回最终的 3 部。

第二步：将向量距离映射为相似度分数。

```python
similarity = 1.0 / (1.0 + distance)
```

距离越小，对应的相似度越高。

第三步：使用 Min-Max 方法分别归一化相似度和电影评分。

```python
df['norm_sim'] = (
    (df['similarity'] - df['similarity'].min())
    / (df['similarity'].max() - df['similarity'].min())
)
```

实际代码还针对最大值等于最小值的情况进行了处理，避免除零错误。

电影评分采用相同的归一化思想。

第四步：进行加权融合。

```python
df['final_score'] = (
    df['norm_sim'] * 0.7
    + df['norm_rat'] * 0.3
)
```

其中：

- 语义相似度权重为 0.7。
- 电影评分权重为 0.3。

这样的设计让语义相关性仍然是主要排序依据，同时让电影评分参与排序。

第五步：按照综合得分重新排序。

```python
df = df.sort_values(
    by='final_score',
    ascending=False
)

top_df = df.head(n_results)
```

最后返回排名靠前的电影。

### 4. 验证结果

第 9 周代码设计了纯语义检索和融合排序的对照实验。

实验选择“高分冷门文艺片”等问题，分别输出两种检索方式返回的电影及其评分，并计算候选电影的平均分。

后续版本将融合排序整合进实际运行的 MovieRetriever 类。

因此，可以确认项目已经实现了语义相似度与电影评分融合的重排序流程。

但目前没有提供完整、可重复的定量评测结果，不能认定融合排序在所有问题上都优于纯语义检索。

此外，融合排序只能改变结果顺序，无法严格保证电影评分大于某个指定阈值。

对于“评分必须大于 4.5”等硬性要求，后续仍需增加明确的数值过滤逻辑。

### 5. 经验总结

这次问题让我理解了语义检索与业务规则之间的区别。

语义检索能够帮助系统找到内容相关的电影，但无法天然满足评分、年份等精确约束。

因此，实际推荐系统不能只依赖向量相似度，还需要结合业务数据进行过滤或排序。

本次改进采用了“扩大候选池 + 归一化 + 加权重排序”的方式。

这也是项目从基础语义检索向更完整的电影推荐系统演进的重要一步。
