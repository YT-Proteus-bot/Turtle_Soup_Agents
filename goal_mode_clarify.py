# agents_learn_demo/goal_mode_clarify.py
# ==============================================================================
# 目标模式 + 海龟汤澄清 demo。
#
# 在 goal_agent.py 的链路（parse_goal → plan → act → evaluate → synthesize）上，
# 加了两处澄清：
#   1. 前置澄清：parse_goal 生成验收条件后，用 1~2 个是/否问题确认/修正它们；
#   2. 按需澄清：evaluate 判定 blocked 时，如果还有预算，先问用户一个问题，
#      再回到 plan 带着新信息重规划。
#
# 图结构：
#   START → parse_goal → pre_clarify → plan → act → evaluate ─┬─ 完成 → synthesize → END
#                                                              ├─ 阻塞且有预算 → ask_user → plan
#                                                              ├─ 阻塞        → plan
#                                                              └─ 继续        → act
#
# 运行：
#   .\.venv\Scripts\python.exe agents_learn_demo\goal_mode_clarify.py --goal "调研 3 个 AI Agent 框架并输出对比报告"
# ==============================================================================

from __future__ import annotations

import argparse
import os
from typing import Callable, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
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

MAX_PRE_QUESTIONS = 2
PRE_CONFIDENCE = 0.9
MAX_CLARIFY = 2
MAX_ITERATIONS = 8
RETRY_ATTEMPTS = 3

# ============ 结构化输出模型 ============
class GoalSpec(BaseModel):
    refined_goal: str = Field(min_length=5, max_length=500)
    acceptance_criteria: list[str] = Field(min_length=1, max_length=8)


class CriteriaClarification(BaseModel):
    """确认验收条件时的结构化结果。"""

    confidence: float = Field(ge=0.0, le=1.0, description="对验收条件的把握度")
    understanding: str = Field(min_length=1, max_length=300)
    acceptance_criteria: list[str] = Field(default_factory=list, description="确认后的验收条件")
    question: Optional[str] = Field(default=None, description="下一个是/否问题；已确认时留空")


class SubGoalPlan(BaseModel):
    subgoals: list[str] = Field(min_length=1, max_length=6)


class StepExecution(BaseModel):
    result: str = Field(min_length=1, max_length=4000)


class Evaluation(BaseModel):
    progress: float = Field(ge=0.0, le=1.0)
    blocked: bool = False
    blocked_reason: str = Field(default="", max_length=300)
    all_met: bool = False
    reason: str = Field(default="", max_length=500)
    clarify_question: str = Field(default="", description="blocked 时，若要问用户，给出一个是/否问题")


# ============ 状态 ============
class SubGoal(TypedDict):
    id: str
    title: str
    done: bool
    result: str


class GoalClarifyState(TypedDict):
    goal: str
    refined_goal: str
    criteria: list[str]
    subgoals: list[SubGoal]
    current_id: str
    completed_results: list[str]
    progress: float
    blocked: bool
    blocked_reason: str
    report: str
    iterations: int
    max_iterations: int
    clarify_history: list[dict]
    clarify_count: int
    clarify_question: str


# ============ 用户输入 ============
ASK: Callable[[str], str] = input
EXIT_WORDS = {"/exit", "exit", "quit", "q"}
DONE_WORDS = {"/done", "done", "直接开始", "开始"}


class UserExit(Exception):
    pass


def normalize_answer(text: str) -> str:
    value = text.strip()
    lowered = value.lower()
    if lowered in {"是", "对", "嗯", "是的", "没错", "y", "yes", "yeah"}:
        return "是"
    if lowered in {"否", "不", "不是", "没有", "不对", "n", "no", "nope"}:
        return "否"
    return f"补充说明：{value}"


def ask_user(prompt: str) -> str:
    raw = ASK(prompt).strip()
    if raw.lower() in EXIT_WORDS:
        raise UserExit
    if raw.lower() in DONE_WORDS:
        return "用户要求按当前理解继续"
    return normalize_answer(raw)


# ============ 通用 LLM 调用 ============
_STRUCTURED_CACHE: dict[type[BaseModel], object] = {}


def _structured(schema: type[BaseModel]):
    if schema not in _STRUCTURED_CACHE:
        _STRUCTURED_CACHE[schema] = llm.with_structured_output(schema, method="function_calling")
    return _STRUCTURED_CACHE[schema]


def call_structured(schema: type[BaseModel], messages: list) -> BaseModel:
    last_error: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            result = _structured(schema).invoke(messages)
            if result is None:
                raise ValueError("模型没有返回结构化结果")
            return result
        except Exception as exc:
            last_error = exc
            print(f"  [重试] {schema.__name__} 第 {attempt}/{RETRY_ATTEMPTS} 次失败：{type(exc).__name__}: {exc}")
    raise RuntimeError(f"{schema.__name__} 连续 {RETRY_ATTEMPTS} 次失败") from last_error


def call_text(messages: list) -> str:
    resp = llm.invoke(messages)
    content = resp.content
    return content.strip() if isinstance(content, str) else str(content).strip()


# ============ 节点 1：parse_goal ============
def parse_goal(state: GoalClarifyState) -> dict:
    print("\n[ParseGoal] 解析目标...")
    output: GoalSpec = call_structured(
        GoalSpec,
        [
            SystemMessage(content="你是目标分析器。重述用户目标，并给出可验证的验收条件。"),
            HumanMessage(content=f"用户目标：{state['goal']}"),
        ],
    )
    print(f"  精炼目标：{output.refined_goal}")
    print(f"  验收条件：{output.acceptance_criteria}")
    return {"refined_goal": output.refined_goal.strip(), "criteria": output.acceptance_criteria}


# ============ 节点 2：pre_clarify（确认验收条件） ============
def pre_clarify(state: GoalClarifyState) -> dict:
    print("\n[Clarify] 确认验收条件...")
    history: list = [
        HumanMessage(
            content=f"""目标：{state['refined_goal']}
当前验收条件：
{chr(10).join(f'- {c}' for c in state['criteria'])}"""
        )
    ]
    criteria = list(state["criteria"])
    confidence = 0.0

    for _ in range(MAX_PRE_QUESTIONS):
        result: CriteriaClarification = call_structured(
            CriteriaClarification,
            [
                SystemMessage(
                    content=(
                        "你是验收条件审查员。用是/否问题确认验收条件是否准确、可验证、没有遗漏关键项。"
                        "每轮只问一个封闭问题；用户可补充文字。"
                        "确认完成后 confidence>=0.95、question 留空，并在 acceptance_criteria 里给出最终条件。"
                    )
                ),
                *history,
            ],
        )
        confidence = result.confidence
        if result.acceptance_criteria:
            criteria = result.acceptance_criteria
        print(f"  理解：{result.understanding}（{confidence:.0%}）")

        if confidence >= PRE_CONFIDENCE or not result.question:
            break
        print(f"  问：{result.question}")
        answer = ask_user("你（是/否/补充）> ")
        history.append(AIMessage(content=result.question))
        history.append(HumanMessage(content=answer))

    print(f"  最终验收条件：{criteria}")
    return {"criteria": criteria}


# ============ 节点 3：plan ============
def plan(state: GoalClarifyState) -> dict:
    print("\n[Plan] 拆子目标...")
    existing = state["subgoals"]
    done_titles = {sg["title"] for sg in existing if sg["done"]}
    done_text = "\n".join(f"- {sg['title']}：{sg['result'][:100]}" for sg in existing if sg["done"]) or "- 无"

    output: SubGoalPlan = call_structured(
        SubGoalPlan,
        [
            SystemMessage(content="你是任务拆解器。把目标拆成按顺序执行、能产出具体结果的子目标。"),
            HumanMessage(
                content=f"""目标：{state['refined_goal']}
验收条件：
{chr(10).join(f'- {c}' for c in state['criteria'])}

已完成的子目标（不要重复）：
{done_text}

请给出 3-6 个子目标；如果是重规划，只给需要新增的。"""
            ),
        ],
    )

    merged: list[SubGoal] = list(existing)
    existing_titles = {sg["title"] for sg in existing}
    start = len(merged) + 1
    for offset, title in enumerate(output.subgoals):
        title = title.strip()
        if not title or title in existing_titles or title in done_titles:
            continue
        merged.append({"id": f"sg{start + offset}", "title": title, "done": False, "result": ""})

    if not merged:
        merged = [
            {"id": "sg1", "title": f"完成目标：{state['refined_goal']}", "done": False, "result": ""}
        ]

    next_sg = next((sg for sg in merged if not sg["done"]), None)
    current_id = next_sg["id"] if next_sg else "__DONE__"

    for sg in merged:
        mark = "x" if sg["done"] else " "
        print(f"  [{mark}] {sg['id']} {sg['title']}")
    return {"subgoals": merged, "current_id": current_id}


# ============ 节点 4：act ============
def act(state: GoalClarifyState) -> dict:
    current = next((sg for sg in state["subgoals"] if sg["id"] == state["current_id"]), None)
    if current is None:
        return {}

    print(f"\n[Act] 执行：{current['title']}")
    try:
        output: StepExecution = call_structured(
            StepExecution,
            [
                SystemMessage(content="你是执行 Agent。完成指定子目标，输出具体结果，不要空话。"),
                HumanMessage(
                    content=f"""目标：{state['refined_goal']}
验收条件：
{chr(10).join(f'- {c}' for c in state['criteria'])}

当前子目标：{current['title']}

已完成结果：
{chr(10).join(f'- {r[:150]}' for r in state['completed_results']) or '- 无'}

请输出这个子目标的结果。"""
                ),
            ],
        )
        result = output.result.strip()
    except Exception as exc:
        result = f"执行失败：{type(exc).__name__}: {exc}"
        print(f"  [降级] {exc}")

    updated: list[SubGoal] = []
    for sg in state["subgoals"]:
        if sg["id"] == current["id"]:
            updated.append({**sg, "done": True, "result": result})
        else:
            updated.append(sg)

    next_sg = next((sg for sg in updated if not sg["done"]), None)
    next_id = next_sg["id"] if next_sg else "__DONE__"
    print(f"  结果预览：{result[:100]}")
    return {
        "subgoals": updated,
        "completed_results": list(state["completed_results"]) + [result],
        "current_id": next_id,
        "iterations": state["iterations"] + 1,
    }


# ============ 节点 5：evaluate ============
def evaluate(state: GoalClarifyState) -> dict:
    all_done = all(sg["done"] for sg in state["subgoals"])
    done_titles = [sg["title"] for sg in state["subgoals"] if sg["done"]]

    print("\n[Evaluate] 评估进度...")
    output: Evaluation = call_structured(
        Evaluation,
        [
            SystemMessage(
                content=(
                    "你是质量评估员。判断进度、是否被阻塞、验收条件是否全部满足。"
                    "如果被阻塞且需要用户提供信息才能继续，在 clarify_question 里给出一个"
                    "能用是/否回答的问题。"
                )
            ),
            HumanMessage(
                content=f"""目标：{state['refined_goal']}
验收条件：
{chr(10).join(f'- {c}' for c in state['criteria'])}

已完成子目标：
{chr(10).join(f'- {t}' for t in done_titles) or '- 无'}

已完成结果摘要：
{chr(10).join(f'- {r[:150]}' for r in state['completed_results'][-5:]) or '- 无'}

请评估。"""
            ),
        ],
    )

    done_flag = output.all_met or all_done
    print(f"  进度：{output.progress * 100:.0f}%  阻塞：{output.blocked}  验收：{output.all_met}")
    return {
        "progress": output.progress,
        "blocked": output.blocked,
        "blocked_reason": output.blocked_reason,
        "clarify_question": output.clarify_question.strip(),
        "current_id": "__DONE__" if done_flag else state["current_id"],
    }


# ============ 节点 6：ask_user（按需澄清） ============
def ask_user_node(state: GoalClarifyState) -> dict:
    question = state["clarify_question"] or "当前信息不足，是否继续用已有信息尝试？"
    print("\n[Clarify] 执行中被阻塞，需要确认：")
    print(f"  问：{question}")
    answer = ask_user("你（是/否/补充）> ")

    history = list(state["clarify_history"])
    history.append({"question": question, "answer": answer})
    print()
    return {
        "clarify_history": history,
        "clarify_count": state["clarify_count"] + 1,
        "clarify_question": "",
        "blocked": False,
    }


# ============ 节点 7：synthesize ============
def synthesize(state: GoalClarifyState) -> dict:
    print("\n[Synthesize] 生成最终报告...")
    results = "\n".join(
        f"## {sg['title']}\n{sg['result']}" for sg in state["subgoals"] if sg["result"]
    ) or "（没有子目标结果）"
    clarifications = "\n".join(
        f"- {item['question']} → {item['answer']}" for item in state["clarify_history"]
    ) or "- 无"

    report = call_text(
        [
            SystemMessage(content="你是报告撰写者。基于子目标结果生成最终报告，不要编造。"),
            HumanMessage(
                content=f"""目标：{state['refined_goal']}
验收条件：
{chr(10).join(f'- {c}' for c in state['criteria'])}

子目标结果：
{results}

澄清记录：
{clarifications}

请生成最终报告。"""
            ),
        ]
    )
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60 + "\n")
    return {"report": report}


# ============ 路由 ============
def route_after_plan(state: GoalClarifyState) -> str:
    return "synthesize" if state["current_id"] == "__DONE__" else "act"


def route_after_evaluate(state: GoalClarifyState) -> str:
    if state["current_id"] == "__DONE__":
        return "synthesize"
    if state["iterations"] >= state["max_iterations"]:
        print(f"[预算] 达到最大执行次数 {state['max_iterations']}，收尾。")
        return "synthesize"
    if state["blocked"]:
        if state["clarify_count"] < MAX_CLARIFY and state["clarify_question"]:
            return "ask_user"
        return "plan"
    return "act"


def build_graph():
    builder = StateGraph(GoalClarifyState)
    builder.add_node("parse_goal", parse_goal)
    builder.add_node("pre_clarify", pre_clarify)
    builder.add_node("plan", plan)
    builder.add_node("act", act)
    builder.add_node("evaluate", evaluate)
    builder.add_node("ask_user", ask_user_node)
    builder.add_node("synthesize", synthesize)

    builder.add_edge(START, "parse_goal")
    builder.add_edge("parse_goal", "pre_clarify")
    builder.add_edge("pre_clarify", "plan")
    builder.add_conditional_edges(
        "plan", route_after_plan, {"act": "act", "synthesize": "synthesize"}
    )
    builder.add_edge("act", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"act": "act", "plan": "plan", "ask_user": "ask_user", "synthesize": "synthesize"},
    )
    builder.add_edge("ask_user", "plan")
    builder.add_edge("synthesize", END)
    return builder.compile()


# ============ CLI ============
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="目标模式 + 海龟汤澄清")
    parser.add_argument("--goal", help="目标描述；不传则交互输入")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args(argv)

    goal = (args.goal or "").strip()
    if not goal:
        try:
            goal = input("目标 > ").strip()
        except (EOFError, KeyboardInterrupt):
            return 1
    if not goal:
        return 2

    print("=" * 60)
    print("目标模式 + 海龟汤澄清")
    print("=" * 60)
    print(f"目标：{goal}\n")

    graph = build_graph()
    initial: GoalClarifyState = {
        "goal": goal,
        "refined_goal": "",
        "criteria": [],
        "subgoals": [],
        "current_id": "",
        "completed_results": [],
        "progress": 0.0,
        "blocked": False,
        "blocked_reason": "",
        "report": "",
        "iterations": 0,
        "max_iterations": max(1, args.max_iterations),
        "clarify_history": [],
        "clarify_count": 0,
        "clarify_question": "",
    }

    try:
        final = graph.invoke(initial, config={"recursion_limit": 40})
    except UserExit:
        print("\n已退出（未生成最终报告）")
        return 0
    except Exception as exc:
        print(f"\n[错误] {type(exc).__name__}: {exc}")
        return 1

    print(f"执行子目标：{final['iterations']}；按需澄清次数：{final['clarify_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
