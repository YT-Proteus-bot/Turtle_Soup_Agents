# agents_learn_demo/intent_cache_clarifier.py
# ==============================================================================
# 缓存引导的意图澄清 Agent（CLI demo）。
#
# 在 intent_clarifier.py 的基础上加了一件事：
#   用户开口后，先从历史缓存里检索相似记录；
#   如果命中候选，澄清问题会优先用来“区分候选”；
#   收敛到某条记录时，向用户确认是否复用，而不是从零重做。
#
# 流程：
#   用户输入 → 检索缓存候选
#        ├─ 有候选 → 带候选做澄清（每轮是/否 / 补充）→ 确认复用 → 复用或回退
#        └─ 无候选 → 普通开放澄清 → 生成新摘要 → 写入本次会话缓存
#
# 运行：
#   .\.venv\Scripts\python.exe agents_learn_demo\intent_cache_clarifier.py
#
# 命令：
#   /done   按当前理解收敛
#   /exit   退出
# ==============================================================================

from __future__ import annotations

import os
from datetime import date
from typing import Callable, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# ============ 模型 ============
API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://api.deepseek.com/v1",
    temperature=0,
)

CONFIDENCE_THRESHOLD = 0.95
MAX_ROUNDS = 10
CACHE_TOP_K = 3
CACHE_MIN_SCORE = 0.08     # 相似度低于这个值就不进入缓存分支

# ============ 模拟历史缓存 ============
# 真实系统里这应该是向量库 + 元数据；这里用内存列表 + 字符二元组相似度演示。
class CacheRecord(TypedDict):
    id: str
    title: str
    goal: str
    deliverables: list[str]
    constraints: list[str]
    assumptions: list[str]
    created: str


CACHE: list[CacheRecord] = [
    {
        "id": "rc-001",
        "title": "LangGraph 与 CrewAI 对比表格（面向初学者）",
        "goal": "用 Markdown 表格对比 LangGraph 和 CrewAI，面向初学者，重点比较定位与上手难度",
        "deliverables": ["Markdown 对比表格", "学习建议"],
        "constraints": ["只比较这两个框架", "面向初学者", "重点：定位、上手难度"],
        "assumptions": ["不需要深入源码分析"],
        "created": "2026-09-12",
    },
    {
        "id": "rc-002",
        "title": "三个 AI Agent 框架的对比报告",
        "goal": "调研 LangGraph、AutoGen、CrewAI 三个框架并输出对比报告",
        "deliverables": ["对比报告"],
        "constraints": ["覆盖三个框架", "包含定位差异、适用场景、开发体验、社区活跃度"],
        "assumptions": ["以公开资料为准"],
        "created": "2026-08-03",
    },
    {
        "id": "rc-003",
        "title": "Python 学习路线（分阶段）",
        "goal": "为有编程基础的人设计一份分阶段的 Python 学习路线",
        "deliverables": ["学习路线图", "每阶段练习建议"],
        "constraints": ["分阶段", "面向有基础的学习者"],
        "assumptions": ["每周可投入 8 小时"],
        "created": "2026-07-21",
    },
    {
        "id": "rc-004",
        "title": "周报模板与本周工作总结",
        "goal": "生成本周工作周报，包含进展、问题、下周计划",
        "deliverables": ["周报文档"],
        "constraints": ["三段式结构", "语言简洁"],
        "assumptions": ["按团队统一模板"],
        "created": "2026-10-18",
    },
]


# ============ 相似度检索 ============
def _normalize(text: str) -> str:
    """去掉空白和常见标点，统一小写，便于做字符级相似度。"""
    keep = []
    for char in text.lower():
        if char.isalnum():
            keep.append(char)
    return "".join(keep)


def _bigrams(text: str) -> set[str]:
    return {text[i:i + 2] for i in range(len(text) - 1)}


def similarity(query: str, record: CacheRecord) -> float:
    """用字符二元组的 Dice 系数衡量相似度（对中文不需要分词）。"""
    a = _normalize(query)
    b = _normalize(record["title"] + record["goal"] + "".join(record["constraints"]))
    grams_a, grams_b = _bigrams(a), _bigrams(b)
    if not grams_a or not grams_b:
        return 0.0
    return 2 * len(grams_a & grams_b) / (len(grams_a) + len(grams_b))


def retrieve(query: str) -> list[tuple[float, CacheRecord]]:
    """返回 top-k 候选；低于阈值的直接丢弃。"""
    scored = [(similarity(query, record), record) for record in CACHE]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(score, record) for score, record in scored[:CACHE_TOP_K] if score >= CACHE_MIN_SCORE]


# ============ 结构化输出模型 ============
class Clarification(BaseModel):
    """每轮澄清结果：当前理解 + 把握度 + 候选匹配 + 下一个问题。"""

    confidence: float = Field(ge=0.0, le=1.0, description="对用户意图的把握度 0-1")
    understanding: str = Field(min_length=1, max_length=300, description="目前理解到的意图")
    missing: list[str] = Field(default_factory=list, description="仍不确定的关键点")
    candidate_id: Optional[str] = Field(
        default=None,
        description="如果需求实质等同于某个缓存候选，填它的 id；否则留空",
    )
    question: Optional[str] = Field(
        default=None,
        description="下一个封闭式问题；把握度达到 0.95 或已匹配候选时留空",
    )


class IntentSummary(BaseModel):
    goal: str = Field(min_length=1, description="一句话说清目标")
    deliverables: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ============ 提示词 ============
BASE_RULES = f"""你是意图澄清员，用“海龟汤”式提问把需求问清楚。

规则：
1. 每轮只问一个能用“是/否”回答的单一判断；用户可以只回“是/否”，也可以补充文字。
2. 优先问最能区分不同意图的问题，不要一次问多个点，也不要重复已确认的信息。
3. 当你能完整复述目标、对象、约束、交付形式时，confidence 给到 {CONFIDENCE_THRESHOLD} 以上，question 留空。
4. 不要虚高：还有关键歧义时 confidence 必须低于 {CONFIDENCE_THRESHOLD}。
5. understanding 用一句话概括当前理解，让用户能看出你有没有跑偏。
"""

CACHE_RULES = f"""本轮有历史缓存候选（按相似度排序）。你可以利用它们加快澄清：
- 如果用户需求与某条记录实质相同，把 candidate_id 设成它的 id；
  当 candidate_id 有值且 confidence >= {CONFIDENCE_THRESHOLD} 时，question 留空，系统会向用户确认是否复用。
- 只有当目标、范围、约束、交付形式都与记录一致时才设 candidate_id；
  用户新增了记录里没有的要求（新维度、新格式、新限制）时，candidate_id 必须留空。
- 如果用户需求只是部分相似，candidate_id 留空，并问“最能区分候选”的封闭问题。
- 绝不为了命中缓存而牵强匹配；用户明确否认候选后，不要再提它。
"""

SUMMARY_SYSTEM = """你是需求整理员。根据澄清对话输出最终意图摘要：
goal / deliverables / constraints / assumptions。
不要编造对话里没出现过的要求；不确定的内容放进 assumptions。
"""

_clarifier = llm.with_structured_output(Clarification, method="function_calling")
_summarizer = llm.with_structured_output(IntentSummary, method="function_calling")

# ============ 回答归一化 ============
YES_WORDS = {"是", "对", "嗯", "是的", "没错", "可以", "y", "yes", "yeah"}
NO_WORDS = {"否", "不", "不是", "没有", "不对", "n", "no", "nope"}
EXIT_WORDS = {"/exit", "exit", "quit", "q"}
DONE_WORDS = {"/done", "done", "直接开始", "开始"}


def normalize_answer(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered in YES_WORDS:
        return "是"
    if lowered in NO_WORDS:
        return "否"
    return f"补充说明：{stripped}"


def format_candidates(scored: list[tuple[float, CacheRecord]]) -> str:
    lines = []
    for score, record in scored:
        lines.append(
            f"[{record['id']}] {record['title']}（相似度 {score:.2f}，创建于 {record['created']}）\n"
            f"    目标：{record['goal']}"
        )
    return "\n".join(lines)


# ============ LLM 调用 ============
def next_clarification(
    history: list,
    candidates: list[tuple[float, CacheRecord]],
    notice: str = "",
) -> Clarification:
    blocks = [BASE_RULES]
    if candidates:
        blocks.append(CACHE_RULES)
        blocks.append("缓存候选：\n" + format_candidates(candidates))
    if notice:
        blocks.append(notice)
    return _clarifier.invoke([SystemMessage(content="\n".join(blocks)), *history])


def build_summary(history: list, confidence: float) -> IntentSummary:
    summary = _summarizer.invoke(
        [
            SystemMessage(content=SUMMARY_SYSTEM),
            *history,
            HumanMessage(content=f"澄清结束，最终把握度约 {confidence:.0%}，请输出意图摘要。"),
        ]
    )
    summary.confidence = max(0.0, min(1.0, confidence))
    return summary


# ============ 缓存复用 ============
def record_to_summary(record: CacheRecord) -> IntentSummary:
    return IntentSummary(
        goal=record["goal"],
        deliverables=list(record["deliverables"]),
        constraints=list(record["constraints"]),
        assumptions=list(record["assumptions"]),
        confidence=1.0,
    )


def add_to_cache(summary: IntentSummary) -> CacheRecord:
    """把新澄清出的意图写入本次会话缓存（仅内存）。"""
    record: CacheRecord = {
        "id": f"new-{len(CACHE) + 1:03d}",
        "title": summary.goal[:30],
        "goal": summary.goal,
        "deliverables": list(summary.deliverables),
        "constraints": list(summary.constraints),
        "assumptions": list(summary.assumptions),
        "created": date.today().isoformat(),
    }
    CACHE.append(record)
    return record


# ============ 主流程 ============
def run_clarification(
    user_input: str,
    ask: Callable[[str], str] = input,
    max_rounds: int = MAX_ROUNDS,
) -> Optional[tuple[IntentSummary, Optional[CacheRecord]]]:
    """返回 (意图摘要, 复用的缓存记录)；/exit 时返回 None。"""
    scored = retrieve(user_input)
    if scored:
        print("\n[缓存] 找到相似的历史记录：")
        for score, record in scored:
            print(f"  - [{record['id']}] {record['title']}（相似度 {score:.2f}）")
    else:
        print("\n[缓存] 没有可用的历史记录，走开放澄清。")

    history: list = [HumanMessage(content=user_input)]
    confidence = 0.0
    notice = ""
    candidates = scored

    for round_no in range(1, max_rounds + 1):
        clarification = next_clarification(history, candidates, notice)
        confidence = clarification.confidence

        print(f"\n[第{round_no}轮] 当前理解：{clarification.understanding}")
        print(f"          把握度：{confidence:.0%}")
        if clarification.candidate_id:
            print(f"          匹配候选：{clarification.candidate_id}")
        if clarification.missing:
            print(f"          待确认：{'；'.join(clarification.missing[:3])}")

        # 命中候选且把握度够 → 让用户确认是否复用
        if clarification.candidate_id and confidence >= CONFIDENCE_THRESHOLD:
            record = next((r for _, r in candidates if r["id"] == clarification.candidate_id), None)
            if record:
                print(f"\n[复用确认] 这条需求与历史记录 [{record['id']}] 相似：")
                print(f"  标题：{record['title']}")
                print(f"  创建于：{record['created']}（请注意是否已过期）")
                answer = ask("是否复用该历史结果？(是/否/补充) > ").strip()
                if answer.lower() in EXIT_WORDS:
                    return None
                if answer.lower() in YES_WORDS:
                    return record_to_summary(record), record
                # 否认或补充：把这次确认写入历史，然后回到开放澄清
                print("[复用确认] 用户没有接受复用，改为按全新需求澄清。")
                history.append(AIMessage(content=f"是否复用历史记录 {record['id']}（{record['title']}）？"))
                history.append(HumanMessage(content=normalize_answer(answer) if answer else "否"))
                candidates = []
                notice = "用户刚刚否认了缓存匹配，请忽略之前的候选，按全新需求继续澄清。"
                continue

        if confidence >= CONFIDENCE_THRESHOLD:
            print(f"\n[完成] 把握度 {confidence:.0%}，停止提问。")
            break

        question = clarification.question or "还有其他需要补充的要求吗？"
        print(f"\n问：{question}")
        try:
            raw = ask("你（是/否/补充）> ")
        except (EOFError, KeyboardInterrupt):
            print("\n已取消")
            return None

        answer = raw.strip()
        if not answer:
            print("（空回答，跳过本轮）")
            continue
        if answer.lower() in EXIT_WORDS:
            print("已退出")
            return None
        if answer.lower() in DONE_WORDS:
            print(f"\n[跳过] 按当前理解收敛（把握度 {confidence:.0%}）。")
            break

        history.append(AIMessage(content=question))
        history.append(HumanMessage(content=normalize_answer(answer)))
    else:
        print(f"\n[提示] 已问满 {max_rounds} 轮，按当前理解收敛。")

    summary = build_summary(history, confidence)
    record = add_to_cache(summary)
    print(f"\n[缓存] 已把本次意图写入会话缓存（{record['id']}）。")
    return summary, None


def print_summary(summary: IntentSummary, reused: Optional[CacheRecord]) -> None:
    print("\n" + "=" * 60)
    print("意图确认" + (f"（复用 {reused['id']}）" if reused else ""))
    print("=" * 60)
    print(f"目标：{summary.goal}")
    for title, items in (
        ("交付物", summary.deliverables),
        ("约束", summary.constraints),
        ("假设", summary.assumptions),
    ):
        print(f"{title}：")
        if items:
            for item in items:
                print(f"  - {item}")
        else:
            print("  - 无")
    print(f"把握度：{summary.confidence:.0%}")
    print("=" * 60)


def main() -> int:
    print("=" * 60)
    print("缓存引导的意图澄清 Agent")
    print("回答方式：是 / 否 / 直接输入补充说明")
    print("命令：/done 直接收敛   /exit 退出")
    print("=" * 60)

    try:
        user_input = input("\n请描述你的需求 > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消")
        return 1

    if not user_input:
        print("需求不能为空")
        return 2
    if user_input.lower() in EXIT_WORDS:
        return 0

    result = run_clarification(user_input)
    if result is None:
        return 0
    summary, reused = result
    print_summary(summary, reused)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
