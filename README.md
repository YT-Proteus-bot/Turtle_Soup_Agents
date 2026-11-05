# 海龟汤 Agent 系列（Turtle Soup Agents）

用二元提问把不确定性收敛到阈值。

这个目录里有十份同源的单文件 demo：五份基础变体，两份把澄清嵌进「计划模式」「目标模式」的组合 demo，外加 RAG 场景验证、层级协作、平级协作各一份。它们共享同一个思路：**在行动之前，先用「是 / 否」问题把不确定性压到 95% 以下**。区别只在于——问谁、在哪个阶段问、由谁判定「够了」。

> 说明：本 README 介绍这批海龟汤系列文件。目录里的 `react*.py`、`plan*.py`、`goal*.py` 是另一组独立 demo，不属于本系列。

## 为什么叫「海龟汤」

海龟汤是一种只能回答「是 / 否」的推理游戏：通过不断提出封闭问题，逐步排除不可能，逼近唯一真相。

Agent 里的很多错误不是因为模型不会做，而是因为**信息不够就开始做**。海龟汤式澄清就是把「够不够」变成一个显式的、可判定的步骤：每问一轮，候选空间缩小一次；达到把握度阈值再行动。

## 共同设计原则

| 原则 | 具体做法 |
| --- | --- |
| 二元问题 | 每轮只问一个能用「是 / 否」回答的单一判断，同时允许用户补充文字 |
| 把握度门控 | 默认阈值 `0.95`，没到就继续澄清；到了才允许进入下一步 |
| 轮次预算 | 每个 demo 都有 `MAX_ROUNDS` 或类似的预算，防止无限追问 |
| 逃生口 | `/done` 按当前理解直接收敛，`/exit` 退出；候选场景允许「都不是」 |
| 结构化输出 | 用 Pydantic + `with_structured_output(method="function_calling")`，不靠解析自由文本 |
| 结论可复核 | 收尾产出结构化摘要或带证据的结论，而不是一段模糊的自然语言 |

## demo 一览

| 文件 | 别名 | 问谁 | 问什么 | 判定依据 | 收尾产物 |
| --- | --- | --- | --- | --- | --- |
| `intent_clarifier.py` | 意图汤 | 用户 | 目标、范围、交付形式 | 模型自评把握度 | 意图摘要 |
| `intent_cache_clarifier.py` | 缓存汤 | 用户 + 历史缓存 | 候选之间的判别特征 | 候选匹配 + 把握度 | 复用历史或新摘要 |
| `observation_clarifier.py` | 观察汤（问人） | 用户 | 哪些观察算数、意味着什么 | 模型自评把握度 | 观察结论 + 模拟动作 |
| `observation_gate.py` | 闸门汤（独立检查） | 检查节点 | 本轮观察够不够回答 | 独立 `check_observation` 节点 | 回答或带缺口重查 |
| `observation_reflect.py` | 反思汤（自检） | 自己 | 五类二元检查项 | **代码按清单算分** | 回答或带缺口重查 |
| `plan_mode_clarify.py` | 计划汤 | 用户（前置 + 按需） | 任务边界 / 计划歧义 | 把握度 + replanner 触发 | 计划执行结果 |
| `goal_mode_clarify.py` | 目标汤 | 用户（前置 + 按需） | 验收条件 / 阻塞点 | 把握度 + evaluate 触发 | 目标报告 |
| `rag_clarify.py` | RAG 汤 | 用户（检索前 + 检索后） | 年份/地区范围、召回是否足够 | 把握度 + 检索结果判定 | 带引用的回答 |
| `multi_agent_clarify.py` | 协作汤 | 组织者 / 员工 / 审查者 | 能力、范围、验收 | 结构化问答 + 未知升级 | 最终结论 + 协作记录 |
| `peer_agent_clarify.py` | 平级汤 | agent 之间（无组织者） | 接口、证据、验收标准 | 证据优先 + 第三方仲裁 | 最终结论 + 协作黑板 |

## 链路对比

```
意图汤
  用户输入 → 澄清循环（每轮 1 个是/否问题）→ 把握度 ≥95% → 意图摘要

缓存汤
  用户输入 → 检索历史缓存 → 带候选澄清 → 命中候选 → 用户确认 → 复用
                                   └─ 未命中/否认 → 开放澄清 → 新摘要入缓存

观察汤（问人）
  工具返回告警 → 问用户"哪些算数" → 把握度 ≥95% → 观察结论 → 确认告警

闸门汤（独立检查）
  plan_action → tools → check_observation ─┬─ 够 → answer
                                           └─ 不够 → 带着 next_focus 重查

反思汤（自检）
  plan_action → tools → observe（草稿 + 自检 + 代码算分）─┬─ ≥95% → answer
                                                          └─ 不够 → 重查

计划汤
  pre_clarify → planner → executor → replanner ─┬─ needs_input → ask_user → replanner
                                                ├─ 还有步骤 → executor
                                                └─ 完成 → finalize

目标汤
  parse_goal → pre_clarify → plan → act → evaluate ─┬─ 完成 → synthesize
                                                    ├─ 阻塞且有预算 → ask_user → plan
                                                    └─ 继续 → act

RAG 汤
  问题 → 检索前澄清（年份/地区）→ 过滤检索 → 检索后判定 ─┬─ 足够 → 生成带引用回答
                                                          └─ 不足 → 再问一轮 → 重检索 → 回答

协作汤
  组织者问用户范围 → 确认员工能力 → 员工提问 ─┬─ 组织者能答 → 回答
                                             └─ 答不上来 → 升级问用户
  → 检索员工取资料 → 写作员工成稿 → 审查者二元验收 ─┬─ 通过 → 交付
                                                    └─ 未过 → 重写一次

平级汤
  Retrieval → Writing → Reviewer（固定顺序，无组织者）
  每人在动手前可向平级提 1 个是/否问题
  回答 unknown → 第三个 agent 仲裁 → 仍 unknown → 升级用户（有次数上限）
  提问总数有预算，用完停止澄清、带缺口继续
```

## 各 demo 说明

### 1. 意图汤 — `intent_clarifier.py`

**一句话**：在动手之前，把「用户到底要什么」问清楚。

- 输入：一句模糊需求，例如「帮我做个 AI 框架的对比」。
- 交互：每轮回答 `是 / 否 / 补充说明`；`/done` 提前收敛，`/exit` 退出。
- 输出：`goal / deliverables / constraints / assumptions / confidence` 五段式意图摘要。
- 关键参数：`CONFIDENCE_THRESHOLD=0.95`、`MAX_ROUNDS=8`。
- 实现要点：把握度分档写进提示词，避免模型每轮只加一点点；明确需求通常 2~3 轮收敛，模糊需求 4~6 轮。

运行：

```powershell
.\.venv\Scripts\python.exe agents_learn_demo\intent_clarifier.py
```

### 2. 缓存汤 — `intent_cache_clarifier.py`

**一句话**：先用历史记录缩小候选，再用问题在候选之间做判别。

- 输入：一句需求，例如「帮我用表格对比 LangGraph 和 CrewAI」。
- 检索：字符二元组的 Dice 相似度（不依赖分词/向量库），取 top-3，低于 `0.08` 丢弃。
- 澄清：有候选时，问题优先用于区分候选；模型可返回 `candidate_id`。
- 复用：候选匹配且把握度 ≥95% 时，展示记录标题与创建日期，由用户确认；否认则回退到开放澄清。
- 输出：复用的历史摘要，或新生成的摘要（并写入本次会话缓存）。
- 关键参数：`CACHE_TOP_K=3`、`CACHE_MIN_SCORE=0.08`、`CONFIDENCE_THRESHOLD=0.95`、`MAX_ROUNDS=6`。
- 防呆：提示词要求「用户新增了记录里没有的要求时，`candidate_id` 必须留空」，避免为了命中缓存而牵强匹配。

运行：

```powershell
.\.venv\Scripts\python.exe agents_learn_demo\intent_cache_clarifier.py
```

### 3. 观察汤（问人）— `observation_clarifier.py`

**一句话**：工具返回一堆告警时，先问用户「哪些才算数」。

- 场景：`list_alerts()` 返回 6 条混合告警（prod/staging、P1~P4、firing/resolved）。
- 交互：每轮 `是 / 否 / 补充`，问题围绕环境、状态、级别、服务、是否同一根因。
- 输出：要处理的告警 ID、观察含义、建议动作，并调用 `acknowledge_alerts()` 模拟动作。
- 关键参数：`CONFIDENCE_THRESHOLD=0.95`、`MAX_ROUNDS=6`。
- 注意：问题必须能从用户那里得到答案；这版默认用户知道自己的告警范围。

运行：

```powershell
.\.venv\Scripts\python.exe agents_learn_demo\observation_clarifier.py
```

### 4. 闸门汤（独立检查）— `observation_gate.py`

**一句话**：不问人，让独立的检查节点判断「本轮观察够不够」。

- 场景：`checkout` 服务排障，工具包括 `status / metrics / errors`。
- 循环：`plan_action`（选工具）→ `tools`（执行）→ `check_observation`（判定）。
- 判定：模型返回 `sufficient / confidence / missing / next_focus / should_ask_user`；达到阈值才收尾。
- 防死循环：最多 `MAX_ROUNDS=5` 轮；`(tool, args)` 完全重复的查询直接 blocked；模型返回 `none` 时交给检查节点判定。
- 输出：最终回答 + 证据；观察不充分时明确写出缺口。

运行：

```powershell
.\.venv\Scripts\python.exe agents_learn_demo\observation_gate.py
# 或
.\.venv\Scripts\python.exe agents_learn_demo\observation_gate.py "checkout 为什么变慢了？"
```

### 5. 反思汤（自检）— `observation_reflect.py`

**一句话**：让 observation 节点先写草稿，再用海龟汤思路对自己做二元检查，把握度由代码算。

- 场景与闸门汤相同，额外提供 `db_pool` 工具（连接池占用、排队、慢查询），用于定位数据库环节。
- 节点内两段式：
  1. `make_draft()` 基于观察写分析草稿（禁止推测观察之外的信息）；
  2. `review_observation()` 生成固定五类检查项：对象状态、关键数据、瓶颈定位、是否回答问题、证据是否充足；每项回答 `yes/no/unknown` 并引用观察原文。
- 把握度（代码计算）：

  ```text
  confidence = Σ(weight × yes) / Σ(weight)

  任一 critical 项 != yes      → 封顶 0.9
  没有 critical 项             → 封顶 0.85
  检查项少于 3 条              → 封顶 0.8
  answer=yes 但 evidence 为空  → 强制降级为 unknown
  ```

- 结论只在 `confidence >= 0.95` 时判定「够了」；否则带着缺口重查，轮数用完则带缺口回答。
- 关键参数：`MAX_ROUNDS=5`、`CONFIDENCE_THRESHOLD=0.95`。

运行：

```powershell
.\.venv\Scripts\python.exe agents_learn_demo\observation_reflect.py
```

### 6. 计划汤 — `plan_mode_clarify.py`

**一句话**：在计划模式的循环里加两处澄清——规划前问边界，replanner 拿不准时问选择。

- 图结构：`pre_clarify → planner → executor → replanner`，replanner 返回 `needs_input` 时走 `ask_user` 再回到 replanner。
- 前置澄清：规划前最多问 `MAX_PRE_QUESTIONS=2` 轮，产出 `constraints / assumptions` 供 planner 使用。
- 按需澄清：replanner 发现「下一步取决于用户的选择」时返回问题；预算 `MAX_CLARIFY=2`，用完自动降级为 `continue`，不会卡死。
- 执行预算：`--max-steps`，默认 6；步骤执行失败会记录为失败步骤并继续。
- 输出：基于已执行步骤的最终回答，并附带澄清记录。

运行：

```powershell
.\.venv\Scripts\python.exe agents_learn_demo\plan_mode_clarify.py --goal "对比 LangGraph 和 CrewAI，输出一份简短对比"
```

### 7. 目标汤 — `goal_mode_clarify.py`

**一句话**：在目标模式里加两处澄清——解析后确认验收条件，evaluate 阻塞时问缺什么。

- 图结构：`parse_goal → pre_clarify → plan → act → evaluate`，阻塞且有预算时走 `ask_user → plan`。
- 前置澄清：`parse_goal` 生成验收条件后，最多问 2 轮是/否问题确认/修正，模型会在 `acceptance_criteria` 里返回最终版本。
- 按需澄清：`evaluate` 判定 blocked 且给出 `clarify_question` 时，先问用户，再把答案带回 `plan` 重规划；预算 `MAX_CLARIFY=2`，用完后直接重规划。
- 执行预算：`--max-iterations`，默认 8。
- 输出：最终报告，并附带澄清记录。

运行：

```powershell
.\.venv\Scripts\python.exe agents_learn_demo\goal_mode_clarify.py --goal "对比 LangGraph 和 CrewAI，输出一份简短对比"
```

### 8. RAG 汤 — `rag_clarify.py`

**一句话**：在检索前后各加一次澄清——先确认范围，再确认召回够不够。

- 场景：本地 mock 语料（差旅、远程办公、采购、信息安全、请假等 7 篇文档），带 `year / region` 元数据。
- 检索：字符覆盖率（50% 单字 + 50% 二元组），支持按年份、地区过滤；过滤后为空会自动放宽并提示。
- 检索前澄清（`QueryClarification`）：确认年份/版本和地区，产出 `filters`；最多 2 轮。
- 检索后判定（`RetrievalCheck`）：判断召回片段是否足够，不够时给出一个关于范围的是/否问题；用户回答后带着补充信息重检索一次，仍不够则带缺口回答。
- 输出：带 `[id]` 引用的回答；回答里会明确说明片段未覆盖的内容。
- 关键参数：`MAX_PRE_QUESTIONS=2`、`PRE_CONFIDENCE=0.9`、`TOP_K=3`。
- 注意：这是流程验证 demo，检索质量本身不是重点；生产环境应换成 embedding + rerank。

运行：

```powershell
.\.venv\Scripts\python.exe agents_learn_demo\rag_clarify.py
```

示例输出（节选）：

```text
问题：出差住宿费能报多少？

[检索前] 是否按 2025 年现行标准？ → 是
[检索前] 是否全国统一标准？     → 是
过滤：year=2025

[检索] d1（0.23）、d3（0.19）、d4（0.05）
[检索后] sufficient=True，confidence=85%

回答：2025 年全国标准：一线城市住宿 600 元/晚、二线 450 元/晚 [d1]；
      片段未覆盖三线城市，也未说明超标处理 [d3]。
```

### 9. 协作汤 — `multi_agent_clarify.py`

**一句话**：组织者、两个员工、一个审查者之间，用结构化的「是/否问答」完成派活、澄清和验收。

- 角色：`Organizer`（拆活、派活、回答与升级）、`Retrieval` 员工（检索）、`Writing` 员工（成稿）、`Reviewer`（验收）。
- 派活前：先查能力清单（`CAPABILITIES`），不浪费模型调用去问“你能做吗”。
- 员工提问：每个员工最多提一个影响执行的封闭问题；组织者只根据任务和已确认约束回答。
- 未知升级：组织者答不上来时 `answer=unknown + escalate=true`，转问用户，而不是自己猜。
- 提问去重：`questions_similar()` 用字符二元组 Jaccard 判断重复，重复问题直接复用已确认结论，避免多个员工反复打断用户。
- 验收：审查者用 3 条二元检查（是否覆盖对象、是否有结论、是否引用资料），不通过让 Writing 带缺口重写一次。
- 输出：最终结论 + 协作记录（谁问了谁、怎么答的）。

运行：

```powershell
.\.venv\Scripts\python.exe agents_learn_demo\multi_agent_clarify.py
```

### 10. 平级汤 — `peer_agent_clarify.py`

**一句话**：没有组织者，三个平级 agent 直接互相提是/否问题，靠证据和仲裁收敛。

- 角色：`Retrieval`（资料）、`Writing`（成稿）、`Reviewer`（验收），彼此没有上下级。
- 提问：每个 agent 在自己动手前，可以向另外两个中的一个提一个封闭问题；问题带原因，回答带证据。
- 回答：`yes / no / unknown`；没有证据只能 `unknown`，不猜。
- 仲裁：`unknown` 时由第三个 agent 按黑板里的证据投票；仍不确定才升级用户。
- 预算：全局最多 `MAX_PEER_QUESTIONS=3` 个问题，用户升级最多 `MAX_USER_ESCALATIONS=1` 次；用尽后停止澄清、按现有信息继续。
- 去重：黑板里已有相似问题时直接复用答案，不重复打扰。
- 输出：最终结论 + 协作黑板（谁问了谁、答案和依据）。

运行：

```powershell
.\.venv\Scripts\python.exe agents_learn_demo\peer_agent_clarify.py
```

#### 多 Agent 协作：两个 demo 怎么选

| 维度 | `multi_agent_clarify.py`（协作汤） | `peer_agent_clarify.py`（平级汤） |
| --- | --- | --- |
| 结构 | 组织者 + 两个员工 + 审查者 | 三个平级 agent，无组织者 |
| 谁回答 | 组织者，按任务和约束回答 | 被问的 agent，按自己的职责回答 |
| 答不上来 | 升级给用户 | 第三方按证据仲裁，仍不确定再升级用户 |
| 防重复 | 组织者侧的问题去重 | 共享黑板 + 问题相似度去重 |
| 预算 | 组织者统一控制 | `MAX_PEER_QUESTIONS` + `MAX_USER_ESCALATIONS` |
| 主要风险 | 组织者独断 | 死锁、互相客套、责任扩散 |
| 适合 | 任务边界清楚、需要统一调度 | 职责互补、接口多、没有天然上级 |

两者共用同一套「是/否 + 证据」协议，区别只在**仲裁权归谁**。

## 怎么选

| 你的情况 | 建议用 |
| --- | --- |
| 需求模糊、还没开始动手 | 意图汤 |
| 业务重复、历史记录多 | 缓存汤 |
| 工具输出需要人来判断相关性 | 观察汤（问人） |
| 想要独立判定、可接受多一次模型调用 | 闸门汤 |
| 想省节点、要可审计的把握度 | 反思汤 |
| 高风险操作、需要人工确认 | 闸门汤 + 人工确认步骤 |
| 高频、低成本场景 | 意图汤 / 闸门汤（控制轮数） |
| 已经在用计划-执行循环，想减少返工 | 计划汤（规划前澄清 + 按需确认） |
| 已经在用目标驱动循环，验收条件容易跑偏 | 目标汤（确认验收 + 阻塞时追问） |
| 做 RAG，检索结果经常答非所问或版本冲突 | RAG 汤（检索前确认范围 + 检索后判定） |
| 多个 agent 分工协作，怕任务歧义或互相甩锅 | 协作汤（能力确认 + 升级机制 + 验收） |
| 多个 agent 平级协作，没有天然仲裁者 | 平级汤（证据优先 + 第三方仲裁 + 预算） |

## 目录文件清单

### 海龟汤系列（本文重点）

前面「demo 一览」表格里的十个文件：

```
intent_clarifier.py          意图汤
intent_cache_clarifier.py    缓存汤
observation_clarifier.py     观察汤（问人）
observation_gate.py          闸门汤（独立检查）
observation_reflect.py       反思汤（自检）
plan_mode_clarify.py         计划汤（计划模式 + 澄清）
goal_mode_clarify.py         目标汤（目标模式 + 澄清）
rag_clarify.py               RAG 汤（检索前后澄清）
multi_agent_clarify.py       协作汤（组织者 + 员工 + 审查者）
peer_agent_clarify.py        平级汤（三个 agent 直接互问）
```

### 同目录的其他 demo（非本系列）

这些是独立示例，和上面的澄清系列没有调用关系，可以单独运行；依赖相同。

| 文件 | 说明 |
| --- | --- |
| `react_agent_cli.py` | CLI 版 ReAct Agent：`create_agent` + Evaluator 验收 + 步数/历史限制 |
| `react.py` | 手写 ReAct 演示：靠提示词约定 `Thought / Action / Action Input`，代码解析文本 |
| `react_langchain.py` | 原生 function calling 版 ReAct：`tool_calls` + `ToolMessage` 回灌 |
| `plan.py` | 计划-执行模式的原始演示（文本解析版） |
| `goals.py` | 目标驱动模式的原始演示（文本解析版） |
| `plan_agent.py` | 生产化计划模式：Pydantic 结构化输出、重试、SQLite 断点续跑 |
| `goal_agent.py` | 生产化目标模式：Pydantic 结构化输出、重试、SQLite 断点续跑 |

> 说明：`react*.py`、`plan.py`、`goals.py`、`plan_agent.py`、`goal_agent.py` 是另一组演示，常用于对比「手写循环 vs 框架封装」「演示版 vs 生产版」，不属于海龟汤澄清系列。

## 环境与依赖

这些脚本共用同一套环境：

- Python 3.11（项目 `.venv`）
- `langchain 1.4.8`、`langchain-core 1.7.4`、`langchain-openai 1.7.3`
- `langgraph 1.2.19`
- `pydantic 2.x`
- 模型：DeepSeek（`https://api.deepseek.com/v1`，`deepseek-chat`）

模型 key 优先读环境变量 `DEEPSEEK_API_KEY`；没设置时使用文件内 fallback（仅本地学习用）。模型名可用 `DEEPSEEK_MODEL` 覆盖。

## 共同实现细节

- **结构化输出**：全部使用 `with_structured_output(..., method="function_calling")`，DeepSeek 支持这一路径。
- **95% 只是启发式阈值**：不是校准过的概率，不同 demo 的「把握度」含义也不同（有的表示理解程度，有的表示观察完整度）。
- **轮次预算**：每个 demo 都有轮次上限，且都处理了「重复提问 / 重复查询」。
- **无持久化**：所有状态都在内存里，进程退出即丢；尚未接 checkpointer。
- **模拟数据**：工具、缓存、告警都是本地 mock，不会产生真实副作用。

## 已知局限

1. **自评偏差**：模型给自己打分容易偏高；反思汤用清单和证据降级来缓解，但没有根除。
2. **提问质量不稳定**：模型可能问出低信息量、重复、甚至不是是/否的问题，提示词只能约束，不能保证。
3. **token 成本随轮数增长**：每轮都重发历史，轮数越多累计输入越大。
4. **用户疲劳**：连续被问 5~7 个问题体验接近审讯，真实产品需要默认值和跳过入口。
5. **缓存可能过期**：缓存汤只展示创建日期提醒，没有自动失效策略。
6. **检索是字面匹配**：字符二元组对同义改写不敏感，生产环境应换成 embedding。
7. **交互依赖终端**：计划汤和目标汤用 `input()` 交互，服务化场景需要改成 interrupt + checkpointer。

## 后续可以做的

- 用「必需槽位覆盖率 + 模型判断」混合计算把握度，替代纯自评。
- 把提问策略从「模型自由发挥」改成「代码决定问哪个槽位，模型只负责措辞」。
- 给澄清阶段接 SQLite checkpointer，支持中断续跑（可参考 `agents/plan_agent.py`）。
- 用 embedding 替换字符相似度，并给缓存加 TTL 与失效原因。
- 建一组固定评测样例，比较这些 demo 的收敛轮数、问题质量、最终答案缺口命中率。
- 把判定模型与生成模型分开（不同模型族），降低自评同源偏差。
