# agents_learn_demo/intent_clarifier.py
# ==============================================================================
# 海龟汤式意图澄清 Agent（CLI demo）。
#
# 思路：先把“用户到底要什么”问清楚，再谈执行。
#   1. 用户给出初始需求；
#   2. Agent 每轮只问一个能用“是/否”回答的封闭问题（类似海龟汤），
#      优先问最能缩小范围的问题；
#   3. 用户回答 是 / 否，也可以直接写一段补充说明；
#   4. Agent 每轮自评“把握度”，达到 95% 就停止提问；
#   5. 最后输出结构化意图摘要（目标、交付物、约束、假设）。
#
# 运行：
#   .\.venv\Scripts\python.exe agents_learn_demo\intent_clarifier.py
#
# 命令：
#   /done   不想再被问了，按当前理解直接收敛
#   /exit   退出
# ==============================================================================

from __future__ import annotations

import os
from typing import Callable, Optional

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

CONFIDENCE_THRESHOLD = 0.95   # 达到这个把握度就停止提问
MAX_ROUNDS = 8                # 最多问多少轮，防止无限追问

# ============ 结构化输出模型 ============
class Clarification(BaseModel):
    """每轮提问的结果：当前理解 + 把握度 + 下一个问题。"""

    confidence: float = Field(ge=0.0, le=1.0, description="当前对用户意图的把握度，0-1")
    understanding: str = Field(min_length=1, max_length=300, description="目前理解到的意图，一句话")
    missing: list[str] = Field(default_factory=list, description="仍然不清楚的关键点，最多 3 条")
    question: Optional[str] = Field(
        default=None,
        description="下一个封闭式问题（能用是/否回答）；把握度达到 0.95 时留空",
    )


class IntentSummary(BaseModel):
    """澄清结束后的最终意图摘要。"""

    goal: str = Field(min_length=1, description="一句话说清用户要达成的目标")
    deliverables: list[str] = Field(default_factory=list, description="需要交付的产物")
    constraints: list[str] = Field(default_factory=list, description="已确认的约束条件")
    assumptions: list[str] = Field(default_factory=list, description="没问出来、但按常理补充的假设")
    confidence: float = Field(ge=0.0, le=1.0, description="最终把握度")


# ============ 提示词 ============
CLARIFY_SYSTEM = f"""你是意图澄清员，用“海龟汤”式提问把用户模糊的需求问清楚。

规则：
1. 每轮只问一个问题，必须能用“是/否”回答；用户可以只回“是/否”，也可以补充一段文字。
2. 问题要具体，优先问最能区分不同意图的问题（类似二分法），不要一次问多个点。
3. 不要重复已经确认过的信息；不要问和任务无关的细节。
4. 当你能完整复述目标、对象、约束、交付形式，且没有关键歧义时，
   confidence 给到 {CONFIDENCE_THRESHOLD} 以上，并把 question 留空。
5. 不要虚高：只要还有关键歧义，confidence 必须低于 {CONFIDENCE_THRESHOLD}，并给出下一个问题。
6. understanding 用一句话概括你现在的理解，让用户能看出你有没有跑偏。

把握度参考（按信息完整度给分，不要机械地每轮只加一点）：
- < 0.5：目标或对象还不明确
- 0.5 ~ 0.8：目标明确，但范围或交付形式仍模糊
- 0.8 ~ 0.94：主体已经清楚，只剩不影响执行的细节
- >= 0.95：目标、对象、约束、交付形式四项都清楚，可以直接执行

如果用户第一句话就已经说清这四项，直接给 >= {CONFIDENCE_THRESHOLD}，不要为了追问而追问。
"""

SUMMARY_SYSTEM = """你是需求整理员。根据澄清对话输出最终意图摘要：
- goal：用户真正要达成的目标
- deliverables：需要交付的产物
- constraints：对话中明确确认的约束
- assumptions：没有确认、但为了可执行而补充的合理假设
不要编造对话里没出现过的要求；不确定的内容放进 assumptions。
"""

# 每个 schema 只构造一次结构化输出 runnable
_clarifier = llm.with_structured_output(Clarification, method="function_calling")
_summarizer = llm.with_structured_output(IntentSummary, method="function_calling")

# ============ 用户回答的归一化 ============
YES_WORDS = {"是", "对", "嗯", "是的", "没错", "y", "yes", "yeah"}
NO_WORDS = {"否", "不", "不是", "没有", "不对", "n", "no", "nope"}
EXIT_WORDS = {"/exit", "exit", "quit", "q"}
DONE_WORDS = {"/done", "done", "直接开始", "开始"}


def normalize_answer(text: str) -> str:
    """把用户输入规范成“是/否/补充说明”。"""
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered in YES_WORDS:
        return "是"
    if lowered in NO_WORDS:
        return "否"
    return f"补充说明：{stripped}"


# ============ 两个 LLM 调用 ============
def next_clarification(history: list) -> Clarification:
    """让模型基于对话历史给出：当前理解、把握度、下一个问题。"""
    return _clarifier.invoke([SystemMessage(content=CLARIFY_SYSTEM), *history])


def build_summary(history: list, confidence: float) -> IntentSummary:
    """澄清结束后生成最终意图摘要。"""
    summary = _summarizer.invoke(
        [
            SystemMessage(content=SUMMARY_SYSTEM),
            *history,
            HumanMessage(content=f"澄清结束，最终把握度约 {confidence:.0%}，请输出意图摘要。"),
        ]
    )
    # confidence 以实际流程的评估为准，避免摘要模型随意填写
    summary.confidence = max(0.0, min(1.0, confidence))
    return summary


# ============ 主流程 ============
def run_clarification(
    user_input: str,
    ask: Callable[[str], str] = input,
    max_rounds: int = MAX_ROUNDS,
) -> Optional[IntentSummary]:
    """执行澄清循环；返回意图摘要，用户 /exit 时返回 None。"""
    history: list = [HumanMessage(content=user_input)]
    confidence = 0.0

    for round_no in range(1, max_rounds + 1):
        clarification = next_clarification(history)
        confidence = clarification.confidence

        print(f"\n[第{round_no}轮] 当前理解：{clarification.understanding}")
        print(f"          把握度：{confidence:.0%}")
        if clarification.missing:
            print(f"          待确认：{'；'.join(clarification.missing[:3])}")

        # 达到阈值 → 停止提问
        if confidence >= CONFIDENCE_THRESHOLD:
            print(f"\n[完成] 把握度达到 {confidence:.0%}，停止提问。")
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
            print(f"\n[跳过] 用户要求直接收敛，当前把握度 {confidence:.0%}。")
            break

        history.append(AIMessage(content=question))
        history.append(HumanMessage(content=normalize_answer(answer)))
    else:
        print(f"\n[提示] 已问满 {max_rounds} 轮，按当前理解收敛（把握度 {confidence:.0%}）。")

    summary = build_summary(history, confidence)
    return summary


def print_summary(summary: IntentSummary) -> None:
    print("\n" + "=" * 60)
    print("意图确认")
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
    print("海龟汤式意图澄清 Agent")
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

    summary = run_clarification(user_input)
    if summary is None:
        return 0
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
