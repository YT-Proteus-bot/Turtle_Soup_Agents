# agents_learn_demo/observation_clarifier.py
# ==============================================================================
# 海龟汤式“观察澄清” Agent（CLI demo）。
#
# 场景：工具返回了一堆告警，但哪些属于本次要处理的、哪些是噪音，
#       从 observation 本身看不出来。Agent 不急着行动，而是像海龟汤一样
#       一轮问一个“是/否”问题，把观察范围收敛到 95% 把握，再决定动作。
#
# 流程：
#   1. 调用工具拿到原始观察（告警列表）；
#   2. 每轮问一个封闭问题（环境？状态？严重级别？服务？）；
#   3. 用户回答 是 / 否 / 补充说明；
#   4. 把握度 >= 95% 时停止，输出观察结论；
#   5. 按结论调用“确认告警”工具，完成一次闭环动作。
#
# 运行：
#   .\.venv\Scripts\python.exe agents_learn_demo\observation_clarifier.py
#
# 命令：
#   /done   直接收敛，按当前理解行动
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

CONFIDENCE_THRESHOLD = 0.95
MAX_ROUNDS = 6

# ============ 模拟环境：告警与动作 ============
# 这些数据就是“世界状态”，Agent 只能通过工具看到它。
ALERTS = [
    {"id": "A-101", "service": "checkout", "env": "prod", "metric": "cpu", "severity": "P1", "status": "firing", "since": "10:02"},
    {"id": "A-102", "service": "checkout", "env": "prod", "metric": "latency", "severity": "P2", "status": "firing", "since": "10:05"},
    {"id": "A-103", "service": "search", "env": "staging", "metric": "cpu", "severity": "P3", "status": "firing", "since": "09:40"},
    {"id": "A-104", "service": "search", "env": "prod", "metric": "disk", "severity": "P2", "status": "resolved", "since": "08:10"},
    {"id": "A-105", "service": "payments", "env": "prod", "metric": "error_rate", "severity": "P1", "status": "firing", "since": "10:01"},
    {"id": "A-106", "service": "frontend", "env": "staging", "metric": "cpu", "severity": "P4", "status": "firing", "since": "09:55"},
]


def list_alerts() -> list[dict]:
    """工具：列出当前所有告警。"""
    return [dict(alert) for alert in ALERTS]


def acknowledge_alerts(alert_ids: list[str]) -> str:
    """工具：确认告警（模拟动作，不产生真实副作用）。"""
    if not alert_ids:
        return "没有选中任何告警，未执行确认动作。"
    known = {alert["id"] for alert in ALERTS}
    valid = [alert_id for alert_id in alert_ids if alert_id in known]
    missing = [alert_id for alert_id in alert_ids if alert_id not in known]
    text = f"已确认告警：{', '.join(valid) if valid else '无'}"
    if missing:
        text += f"；忽略不存在的告警：{', '.join(missing)}"
    return text


def format_alerts(alerts: list[dict]) -> str:
    """把告警列表渲染成可读文本，作为 observation 塞进模型上下文。"""
    lines = [f"{'ID':<7}{'服务':<12}{'环境':<8}{'指标':<12}{'级别':<6}{'状态':<10}开始时间"]
    for a in alerts:
        lines.append(
            f"{a['id']:<7}{a['service']:<12}{a['env']:<8}{a['metric']:<12}"
            f"{a['severity']:<6}{a['status']:<10}{a['since']}"
        )
    return "\n".join(lines)


# ============ 结构化输出模型 ============
class ObservationClarification(BaseModel):
    """每轮观察澄清的结果。"""

    confidence: float = Field(ge=0.0, le=1.0, description="对“哪些告警属于本次要处理的范围、它们意味着什么”的把握度")
    understanding: str = Field(min_length=1, max_length=300, description="目前对观察的理解，一句话")
    relevant_alert_ids: list[str] = Field(default_factory=list, description="当前认为相关的告警 ID")
    missing: list[str] = Field(default_factory=list, description="还不确定的关键点，最多 3 条")
    question: Optional[str] = Field(
        default=None,
        description="下一个封闭式问题（能用是/否回答）；把握度达到 0.95 时留空",
    )


class ObservationConclusion(BaseModel):
    """观察澄清结束后的结论。"""

    selected_alerts: list[str] = Field(default_factory=list, description="最终要处理的告警 ID")
    interpretation: str = Field(min_length=1, description="这些告警说明了什么")
    recommended_action: str = Field(min_length=1, description="建议执行的动作")
    confidence: float = Field(ge=0.0, le=1.0, description="最终把握度")


# ============ 提示词 ============
CLARIFY_SYSTEM = f"""你在做“观察澄清”：工具返回了一份告警列表，但其中哪些告警属于当前要处理的范围、
它们之间是什么关系，仅凭观察本身无法确定。你不能直接假定全部告警都要处理。

你的任务是通过向用户提问来收敛观察，规则：
1. 每轮只问一个能用“是/否”回答的单一判断（例：“是否只处理 prod 环境的告警？”）。
   不要让用户在两个选项里挑，也不要把两个问题用“还是”连在一起。
2. 优先问最能排除候选的问题：环境（prod/staging）、状态（firing/resolved）、
   严重级别（P1/P2）、服务、是否同一根因。
3. 不要问观察里已经写明的事实（例如“A-101 是不是 prod 环境”）。
4. 用户补充的文字视为新的观察约束，要纳入理解。
5. confidence 表示你对“哪些告警要处理、为什么”的把握度，不是对告警数量的把握。
6. 达到 {CONFIDENCE_THRESHOLD} 时停止提问，question 留空。
7. 已经问过、用户也回答过的范围不要再问；每轮回答后 confidence 要按新增信息更新，
   不要连续几轮停在同一个数值。

把握度分档（按信息完整度给，不要每轮只加一点）：
- < 0.5：完全不知道哪些告警相关
- 0.5 ~ 0.8：知道大致范围，但环境/状态/级别还有歧义
- 0.8 ~ 0.94：候选基本确定，只剩细节
- >= 0.95：能明确列出要处理的告警，并解释它们的含义
"""

CONCLUSION_SYSTEM = """你是告警处置员。根据澄清对话输出观察结论：
- selected_alerts：最终要处理的告警 ID（只包含用户确认过或明确属于范围内的）
- interpretation：这些告警合在一起说明了什么
- recommended_action：建议的处置动作，要具体
不要编造对话里没有确认的信息。
"""

_clarifier = llm.with_structured_output(ObservationClarification, method="function_calling")
_concluder = llm.with_structured_output(ObservationConclusion, method="function_calling")

# ============ 用户回答归一化 ============
YES_WORDS = {"是", "对", "嗯", "是的", "没错", "y", "yes", "yeah"}
NO_WORDS = {"否", "不", "不是", "没有", "不对", "n", "no", "nope"}
EXIT_WORDS = {"/exit", "exit", "quit", "q"}
DONE_WORDS = {"/done", "done", "直接处理", "开始处理"}


def normalize_answer(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered in YES_WORDS:
        return "是"
    if lowered in NO_WORDS:
        return "否"
    return f"补充说明：{stripped}"


# ============ LLM 调用 ============
def next_clarification(history: list) -> ObservationClarification:
    return _clarifier.invoke([SystemMessage(content=CLARIFY_SYSTEM), *history])


def build_conclusion(history: list, confidence: float) -> ObservationConclusion:
    conclusion = _concluder.invoke(
        [
            SystemMessage(content=CONCLUSION_SYSTEM),
            *history,
            HumanMessage(content=f"观察澄清结束，最终把握度约 {confidence:.0%}，请给出结论。"),
        ]
    )
    conclusion.confidence = max(0.0, min(1.0, confidence))
    return conclusion


# ============ 主流程 ============
def run_observation_clarification(
    user_request: str,
    ask: Callable[[str], str] = input,
    max_rounds: int = MAX_ROUNDS,
) -> Optional[ObservationConclusion]:
    """先取观察，再澄清，最后给出结论；/exit 时返回 None。"""
    alerts = list_alerts()
    observation = format_alerts(alerts)
    print("\n[观察] 工具返回的原始告警：\n")
    print(observation)
    print()

    history: list = [
        HumanMessage(
            content=f"""用户请求：{user_request}

工具返回的 observation：
{observation}

请判断哪些告警属于本次要处理的范围；如果无法确定，就向用户提问。"""
        )
    ]
    confidence = 0.0

    for round_no in range(1, max_rounds + 1):
        clarification = next_clarification(history)
        confidence = clarification.confidence

        print(f"[第{round_no}轮] 当前理解：{clarification.understanding}")
        print(f"          把握度：{confidence:.0%}")
        if clarification.relevant_alert_ids:
            print(f"          当前候选：{', '.join(clarification.relevant_alert_ids)}")
        if clarification.missing:
            print(f"          待确认：{'；'.join(clarification.missing[:3])}")

        if confidence >= CONFIDENCE_THRESHOLD:
            print(f"\n[完成] 观察把握度达到 {confidence:.0%}，停止提问。")
            break

        question = clarification.question or "还有其他关于告警范围的信息要补充吗？"
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
            print(f"\n[跳过] 用户要求直接处理，当前把握度 {confidence:.0%}。")
            break

        history.append(AIMessage(content=question))
        history.append(HumanMessage(content=normalize_answer(answer)))
    else:
        print(f"\n[提示] 已问满 {max_rounds} 轮，按当前观察收敛（把握度 {confidence:.0%}）。")

    conclusion = build_conclusion(history, confidence)

    # 按结论执行动作（这里是模拟工具）
    action_result = acknowledge_alerts(conclusion.selected_alerts)
    print(f"\n[动作] {action_result}")

    return conclusion


def print_conclusion(conclusion: ObservationConclusion) -> None:
    print("\n" + "=" * 60)
    print("观察结论")
    print("=" * 60)
    print(f"要处理的告警：{', '.join(conclusion.selected_alerts) if conclusion.selected_alerts else '无'}")
    print(f"观察含义：{conclusion.interpretation}")
    print(f"建议动作：{conclusion.recommended_action}")
    print(f"把握度：{conclusion.confidence:.0%}")
    print("=" * 60)


def main() -> int:
    print("=" * 60)
    print("海龟汤式观察澄清 Agent")
    print("场景：一堆告警里，到底哪些需要处理？")
    print("回答方式：是 / 否 / 直接输入补充说明")
    print("命令：/done 直接处理   /exit 退出")
    print("=" * 60)

    try:
        user_request = input("\n请描述你的请求 > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消")
        return 1

    if not user_request:
        print("请求不能为空")
        return 2
    if user_request.lower() in EXIT_WORDS:
        return 0

    conclusion = run_observation_clarification(user_request)
    if conclusion is None:
        return 0
    print_conclusion(conclusion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
