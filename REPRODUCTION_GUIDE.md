# EC Graph 复现指南

本项目当前已经不是单一的“图谱问答 demo”，而是一个围绕电商场景构建的 Agent 原型。系统包含三条主链路：

- 中文商品 NER：从商品标题和描述中抽取 `ATTR`、`PEOPLE`、`SPEC`
- 知识图谱问答：自然语言问题 -> 工具规划 -> 实体对齐 -> Cypher 查询 -> 答案生成
- 多轮手机导购：规则优先的 NLU + 状态管理 + 图约束检索/比较

项目的设计重点不是让 LLM 直接决定商品事实，而是让 LLM 负责语义理解、工具计划和答案表达，商品过滤、价格约束、在售判断和只读查询由图谱与确定性逻辑控制。

## 1. 当前能力概览

### 1.1 Agent Runtime

当前仓库在 `src/agent/` 下已经引入统一运行时：

- `AgentRuntime`：统一工具注册、trace 记录、trace 落盘
- `AgentController`：KGQA 主链路编排
- `ExecutionTrace`：统一执行事件协议
- `CypherGuard`：执行前只读门禁

KGQA 链路的标准工具目前包括：

- `entity_link_tool`
- `graph_query_tool`
- `answer_tool`

导购链路底层也已经接入工具化包装：

- `product_search_tool`
- `product_compare_tool`
- `price_floor_tool`

### 1.2 ExecutionTrace

每次 Agent 运行都会生成一份 `ExecutionTrace`，核心字段来自 [`src/agent/types.py`](src/agent/types.py)：

- `request_id`
- `session_id`
- `user_query`
- `intent`
- `plan`
- `tool_calls`
- `failure_stage`
- `failure_tags`
- `quality_signals`
- `quality_signal_rules`
- `fallback_used`
- `latency_breakdown`
- `final_answer`

当前标准失败标签包括：

- `parse_failure`
- `plan_schema_invalid`
- `entity_missing`
- `entity_misaligned`
- `unsafe_query_blocked`
- `query_timeout`
- `query_empty`
- `answer_weak`

### 1.3 CypherGuard

当前 Cypher 安全门禁位于 [`src/agent/guard.py`](src/agent/guard.py)，策略是“文本粗筛 + `EXPLAIN` 精校”：

- 禁止多语句
- 禁止写操作关键字：`CREATE`、`MERGE`、`SET`、`DELETE`、`REMOVE`、`DROP`
- 禁止 `CALL DBMS`
- 禁止 `LOAD CSV`
- 对 `WHERE` 中的裸字符串字面量做参数化约束
- 自动补齐或裁剪 `LIMIT`，默认上限 `100`
- 可选开启 `EXPLAIN`，把语法/语义校验下推给 Neo4j

说明：

- 这不是 AST 解析器，而是更贴近生产执行口径的只读门禁。
- 单元测试环境如果没有 Neo4j，可使用 `enable_explain=False` 跑文本层规则。

## 2. 代码结构

当前核心目录如下：

```text
src/
|-- agent/
|   |-- controller.py
|   |-- runtime.py
|   |-- guard.py
|   |-- types.py
|   `-- tools/
|       |-- base.py
|       |-- entity_link_tool.py
|       |-- graph_query_tool.py
|       |-- answer_tool.py
|       |-- product_search_tool.py
|       |-- product_compare_tool.py
|       `-- price_floor_tool.py
|-- configuration/
|   |-- config.py
|   `-- entity_normalization.json
|-- datasync/
|   |-- schema_sync.py
|   |-- table_sync.py
|   |-- text_sync.py
|   |-- openbg_sync.py
|   |-- openbg_text_sync.py
|   `-- reset_graph.py
|-- dialogue/
|   |-- nlu.py
|   |-- retrieval.py
|   |-- service.py
|   |-- state.py
|   `-- types.py
|-- eval/
|   |-- dialogue_eval.py
|   |-- dialogue_nlu_eval.py
|   |-- dialogue_smoke.py
|   |-- entity_linking_eval.py
|   `-- kgqa_eval.py
|-- ner/
|   |-- preprocess.py
|   |-- train.py
|   |-- eval.py
|   |-- predict.py
|   `-- normalization.py
`-- web/
    |-- app.py
    |-- memory.py
    |-- schemas.py
    |-- service.py
    `-- static/
```

## 3. 环境准备

### 3.1 Python 与解释器

项目当前约定解释器为：

```powershell
D:\Anaconda_envs\envs\graph\python.exe
```

### 3.2 基础依赖

- Python 3.10+
- MySQL 8.x
- Neo4j 5.x
- 可选：GPU 用于 NER 训练

安装依赖：

```powershell
D:\Anaconda_envs\envs\graph\python.exe -m pip install --upgrade pip
D:\Anaconda_envs\envs\graph\python.exe -m pip install -r requirements.txt
```

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

`.env` 关键项：

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=gmall

NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat
EMBEDDING_MODEL_NAME=BAAI/bge-base-zh-v1.5
```

说明：

- 如果未配置 `DEEPSEEK_API_KEY`，导购接口仍可使用规则 NLU 和固定回复。
- KGQA 的 `/api/chat` 和 `/api/agent/chat` 需要可用的 LLM。
- 实体链接 `hybrid` baseline 和线上 hybrid 对齐依赖 embedding 模型。

## 4. 全量数据链路

### 4.1 NER 预处理、训练与评测

```powershell
D:\Anaconda_envs\envs\graph\python.exe src\ner\preprocess.py
D:\Anaconda_envs\envs\graph\python.exe src\ner\train.py
D:\Anaconda_envs\envs\graph\python.exe src\ner\eval.py
```

`logs/ner/` 下会输出：

- `ner_bad_cases.jsonl`
- `ner_confusion.csv`
- `ner_error_summary.json`

当前实体类型固定为：

```text
ATTR, PEOPLE, SPEC
```

### 4.2 构建主图谱

```powershell
D:\Anaconda_envs\envs\graph\python.exe src\datasync\schema_sync.py
D:\Anaconda_envs\envs\graph\python.exe src\datasync\table_sync.py
D:\Anaconda_envs\envs\graph\python.exe src\datasync\text_sync.py
```

作用：

- `schema_sync.py`：建约束、建索引辅助结构
- `table_sync.py`：同步 gmall 的类目、品牌、SPU、SKU、属性等结构化事实
- `text_sync.py`：对商品文本跑 NER，并写入 `AttributeTag`、`PeopleTag`、`SpecTag`

### 4.3 OpenBG 扩图

```powershell
D:\Anaconda_envs\envs\graph\python.exe src\datasync\openbg_sync.py
D:\Anaconda_envs\envs\graph\python.exe src\datasync\openbg_text_sync.py
```

### 4.4 构建问答索引

```powershell
D:\Anaconda_envs\envs\graph\python.exe src\web\utils.py
```

该步骤会为配置中的节点类型创建全文索引与向量索引，用于实体对齐。

## 5. 评测

评测集位于：

- `data/eval/dialogue_tasks.jsonl`
- `data/eval/dialogue_nlu_samples.jsonl`
- `data/eval/dialogue_smoke_cases.jsonl`
- `data/eval/entity_linking.jsonl`
- `data/eval/kgqa.jsonl`

### 5.1 多轮导购评测

当前导购评测分三层，口径严格分开：

- `dialogue_eval.py`：`StubNLU + FixtureRetriever` 的状态机 / 工具调用回归集
- `dialogue_nlu_eval.py`：真实 `DialogueNLU.parse()` 的单轮诊断集
- `dialogue_smoke.py`：真实 Neo4j + 真实 LLM 的少量 smoke 联调

#### 5.1.1 主评测：状态机 / 工具调用回归集

运行：

```powershell
D:\Anaconda_envs\envs\graph\python.exe src\eval\dialogue_eval.py
```

输入：

- `data/eval/dialogue_tasks.jsonl`

当前任务规模：

- `30` 条任务
- `7` 类覆盖：补槽 / 预算表达 / 品牌存储 / 预算确认 / compare / fallback / reset

主指标：

- `task_success_rate`
- `state_transition_correct_rate`
- `tool_invocation_correct_rate`
- `compare_success_rate`
- `avg_turns_to_success`
- `fallback_rate`
- `recommendation_non_empty_rate`

输出文件：

- `logs/eval/dialogue_offline.jsonl`
- `logs/eval/dialogue_summary.json`

说明：

- 这一层使用 `nlu_outputs` 作为固定输入，评的是状态机与工具调用，不评真实 NLU 抽取质量。
- `expected_tool_counts` 用于校验每轮 `product_search_tool / product_compare_tool / price_floor_tool` 的调用是否符合预期。
- `success_turn` 在数据集中保留为人工注释字段；当前评测逻辑已改为按运行时首次成功终态计算 `avg_turns_to_success`。

#### 5.1.2 NLU 诊断集

运行：

```powershell
D:\Anaconda_envs\envs\graph\python.exe src\eval\dialogue_nlu_eval.py
```

输入：

- `data/eval/dialogue_nlu_samples.jsonl`

当前样本规模：

- `10` 条单轮 NLU 样本

指标：

- `intent_accuracy`
- `slot_f1`
- `fallback_to_llm_rate`

输出文件：

- `logs/eval/dialogue_nlu_cases.jsonl`
- `logs/eval/dialogue_nlu_diagnostic.json`

说明：

- 这一层直接调用真实 `DialogueNLU.parse()`，不进入完整对话循环。
- 如果当前环境缺少真实 LLM 初始化依赖，脚本会输出结构化错误并以非零状态退出。

#### 5.1.3 Smoke 联调

运行：

```powershell
D:\Anaconda_envs\envs\graph\python.exe src\eval\dialogue_smoke.py
```

输入：

- `data/eval/dialogue_smoke_cases.jsonl`

当前样本规模：

- `5` 条 smoke case

输出：

- `logs/eval/dialogue_smoke_<date>.json`

说明：

- 这一层连真实 Neo4j + 真实 LLM，只验证链路打通和 trace 排障。
- 失败时会按同一 `session_id` 聚合导购工具 trace；QA fallback 场景也会把 QA trace 一并归到同一 smoke 日志。

### 5.2 实体链接评测

运行：

```powershell
D:\Anaconda_envs\envs\graph\python.exe src\eval\entity_linking_eval.py
```

可选：

```powershell
D:\Anaconda_envs\envs\graph\python.exe src\eval\entity_linking_eval.py --baseline hybrid --top-k 5
```

支持 baseline：

- `exact_match`
- `fulltext`
- `hybrid`

输出指标：

- `top1_accuracy`
- `topk_recall`
- `by_label_accuracy`

输出文件：

- `logs/eval/entity_linking_<baseline>.jsonl`
- `logs/eval/entity_linking_summary.json`

说明：

- 该评测默认在“已知正确 label”条件下进行。
- `hybrid` 会触发 embedding / vector 检索；`exact_match` 和 `fulltext` 不需要向量模型。

### 5.3 KGQA 分阶段评测

运行：

```powershell
D:\Anaconda_envs\envs\graph\python.exe src\eval\kgqa_eval.py
```

可选：

```powershell
D:\Anaconda_envs\envs\graph\python.exe src\eval\kgqa_eval.py --baseline template
D:\Anaconda_envs\envs\graph\python.exe src\eval\kgqa_eval.py --baseline full
D:\Anaconda_envs\envs\graph\python.exe src\eval\kgqa_eval.py --baseline ablation
D:\Anaconda_envs\envs\graph\python.exe src\eval\kgqa_eval.py --baseline all --enable-session-memory
```

支持 baseline：

- `template`：手写模板 Cypher，不走 LLM，不走实体对齐
- `full`：LLM 规划 + 实体对齐 + 图查询 + 答案生成
- `ablation`：保留 LLM 原始实体参数，跳过实体对齐

输出指标：

- `raw_json_parse_success_rate`
- `repaired_json_parse_success_rate`
- `cypher_query_present_rate`
- `cypher_execution_success_rate`
- `non_empty_result_rate`
- `answer_keyword_hit_rate`
- `unsafe_cypher_rate`
- `entity_any_coverage_rate`
- `entity_all_coverage_rate`

输出文件：

- `logs/eval/kgqa_<baseline>.jsonl`
- `logs/eval/kgqa_summary.json`
- `logs/eval/kgqa_<baseline>_memory_on.jsonl`
- `logs/eval/kgqa_summary_memory_on.json`

说明：

- `cypher_execution_success_rate` 和 `non_empty_result_rate` 只在 `must_execute=true` 子集上统计。
- `answer_keyword_hit_rate` 只在 `has_answer_keywords=True` 的样本上统计。
- `template` baseline 现在可以独立运行，不再因为 eager embedding 初始化阻塞。

### 5.4 评测环境前提

KGQA 评测集必须和当前 Neo4j 中的事实一致。也就是说：

- 改了 `data/gmall.sql`、OpenBG 原始文件或 NER 原始数据，不会自动刷新 Neo4j
- 复现 KGQA 评测前，应先确认必要的 datasync 脚本已经执行
- 如果更换了 Neo4j 实例，建议先用 `template` baseline 验证图事实和查询模板

## 6. API 说明

### 6.1 旧 QA 接口

```http
POST /api/chat
Content-Type: application/json
```

请求：

```json
{
  "message": "Apple 都有哪些产品？",
  "session_id": null
}
```

响应：

```json
{
  "message": "答案文本",
  "session_id": "session-id"
}
```

特点：

- 保留旧 schema
- 保留基于 `session_id` 的轻量 history 记忆
- 内部已经改为调用 Agent Runtime

### 6.2 新 Agent 接口

```http
POST /api/agent/chat
Content-Type: application/json
```

请求：

```json
{
  "message": "Apple 都有哪些产品？",
  "session_id": null
}
```

默认响应：

```json
{
  "answer": "答案文本",
  "session_id": "session-id",
  "trace_id": "trc_20260516_xxxxxxxx",
  "plan_summary": [
    {"tool": "entity_link_tool"},
    {"tool": "graph_query_tool"},
    {"tool": "answer_tool"}
  ],
  "latency_ms": 123,
  "fallback_used": null,
  "trace": null
}
```

调试模式：

```http
POST /api/agent/chat?debug=true
```

此时响应中会内联完整 `trace`。

### 6.3 Trace 查询

获取单条 trace：

```http
GET /api/agent/traces/{trace_id}
```

按 session 查询：

```http
GET /api/agent/traces?session_id=...
```

trace 文件本地落盘路径：

```text
logs/traces/<date>/<trace_id>.json
```

### 6.4 Replay 接口

```http
POST /api/agent/replay
Content-Type: application/json
```

请求：

```json
{
  "trace_ids": ["trc_20260516_xxxxxxxx"]
}
```

说明：

- 当前实现会按 `trace_id` 读取并返回已落盘 trace
- 这更接近“trace replay 输入集获取接口”
- 它还不是一个带多 runtime 变体对比的完整重放评测框架

### 6.5 导购接口

```http
POST /api/dialogue/chat
Content-Type: application/json
```

请求：

```json
{
  "message": "想买手机，预算 5000，主要拍照",
  "session_id": null
}
```

响应字段：

- `session_id`
- `message`
- `mode`
- `action`
- `state`
- `recommendations`

当前导购槽位：

- `budget_max`
- `use_case`
- `brand`
- `storage`

说明：

- 导购状态机仍在 `dialogue/state.py`
- 但底层 `search / compare / get_min_price` 已经通过 Agent Tool 包装执行

## 7. 开发与验证

### 7.1 运行服务

```powershell
D:\Anaconda_envs\envs\graph\python.exe src\web\app.py
```

### 7.2 单元测试

```powershell
D:\Anaconda_envs\envs\graph\python.exe -m unittest discover -s tests -v
```

当前测试覆盖：

- Web API
- Agent guard
- Agent runtime
- KGQA trace 兼容层
- 导购 NLU / retrieval / service
- 导购评测脚本（主评测 / NLU 诊断 / smoke 工具函数）
- 实体链接评测与 KGQA 评测指标计算
- NER 错误分析相关逻辑

### 7.3 常用命令

```powershell
D:\Anaconda_envs\envs\graph\python.exe src\datasync\reset_graph.py
D:\Anaconda_envs\envs\graph\python.exe src\datasync\schema_sync.py
D:\Anaconda_envs\envs\graph\python.exe src\datasync\table_sync.py
D:\Anaconda_envs\envs\graph\python.exe src\web\app.py
```

语法检查：

```powershell
D:\Anaconda_envs\envs\graph\python.exe -m py_compile src\agent\controller.py src\agent\runtime.py src\agent\guard.py src\web\app.py src\web\service.py src\dialogue\service.py
```

## 8. 已知边界

- 多轮导购当前仍是手机品类 demo，不是全品类 Agent
- QA session 和记忆仍是内存存储，服务重启后会丢失
- `Replay` 接口当前只返回持久化 trace，不做多版本 runtime 真正重跑
- Cypher 安全门禁当前是 `EXPLAIN` 驱动的只读校验，不是独立 AST 解析器
- 导购链路虽然接入了统一 trace，但 session 状态机没有和 QA memory 合并
- 线上 KGQA 仍依赖 LLM 生成查询计划，因此生产环境还应继续加强模板路由、权限控制和查询审计
