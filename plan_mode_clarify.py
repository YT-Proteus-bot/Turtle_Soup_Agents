# agents_learn_demo/plan_mode_clarify.py
# ==============================================================================
# 计划模式 + 海龟汤澄清 demo。
#
# 在 plan_agent.py 的循环（planner → executor → replanner → finalize）上，
# 加了两处澄清：
#   1. 前置澄清：规划前先用 1~2 个是/否问题确认范围与交付形式；
#   2. 按需澄清：replanner 发现“下一步取决于用户选择”时，返回 needs_input，
#      由 ask_user 节点问一个问题，再回到 replanner 继续。
#
# 图结构：
#   START → pre_clarify → planner → executor → replanner ─┬─ needs_input → ask_user → replanner
#                                                          ├─ 还有步骤 → executor
#                                                          └─ 完成     → finalize → END
#
# 运行：
#   .\.venv\Scripts\python.exe agents_learn_demo\plan_mode_clarify.py --goal "调研 3 个 AI Agent 框架并输出对比报告"
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

# ============ 模型配置 ============
API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://api.deepseek.com/v1",
    temperature=0,
)

MAX_PRE_QUESTIONS = 2      # 规划前最多问几轮
PRE_CONFIDENCE = 0.9       # 前置澄清达到多少把握就不再问
MAX_CLARIFY = 2            # 执行中最多触发几次按需澄清
MAX_EXECUTIONS = 6         # 最多执行多少步
RETRY_ATTEMPTS = 3

# ============ 结构化输出模型 ============
class PreClarification(BaseModel):
    """规划前的澄清结果。"""

    confidence: float = Field(ge=0.0, le=1.0, description="对任务边界的把握度")
    understanding: str = Field(min_length=1, max_length=300, description="目前理解到的任务")
    constraints: list[str] = Field(default_factory=list, description="已确认的约束")
    assumptions: list[str] = Field(default_factory=list, description="暂时按常理补充的假设")
    question: Optional[str] = Field(default=None, description="下一个是/否问题；边界清楚时留空")


class PlanStep(BaseModel):
    description: str = Field(min_length=3, max_length=300)


class PlannerOutput(BaseModel):
    steps: list[PlanStep] = Field(min_length=1, max_length=8)


class StepResult(BaseModel):
    summary: str = Field(min_length=1, max_length=500)
    details: str = Field(default="", max_length=4000)


class ReplanDecision(BaseModel):
    action: Literal["continue", "modify", "done", "needs_input"] = Field(description="下一步决策")
    new_steps: list[PlanStep] = Field(default_factory=list)
    question: str = Field(default="", description="action=needs_input 时要问用户的问题")
    reason: str = Field(default="", max_length=500)


# ============ 图状态 ============
class PlanClarifyState(TypedDict):
    goal: str
    constraints: list[str]
    assumptions: list[str]
    clarify_history: list[dict]
    clarify_count: int
    clarify_question: str
    plan: list[dict]
    past_steps: list[dict]
    final_answer: str
    executed: int
    max_executions: int
    replan_action: str


# ============ 用户输入与命令 ============
ASK: Callable[[str], str] = input
EXIT_WORDS = {"/exit", "exit", "quit", "q"}
DONE_WORDS = {"/done", "done", "直接开始", "开始"}


class UserExit(Exception):
    """用户主动退出。"""


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


# ============ 节点 1：pre_clarify ============
def pre_clarify(state: PlanClarifyState) -> dict:
    """规划前问 1~2 个是/否问题，确认范围与交付形式。"""
    print("\n[Clarify] 规划前确认任务边界...")
    history: list = [HumanMessage(content=f"用户任务：{state['goal']}")]
    constraints: list[str] = []
    assumptions: list[str] = []
    confidence = 0.0

    for round_no in range(1, MAX_PRE_QUESTIONS + 1):
        result: PreClarification = call_structured(
            PreClarification,
            [
                SystemMessage(
                    content=(
                        "你是任务澄清员。在制定计划前，用是/否问题补齐关键约束："
                        "范围、对象、交付形式、硬性限制。每轮只问一个封闭问题，"
                        "用户可以回是/否或补充文字。边界已经清楚时 confidence>=0.95 且 question 留空。"
                    )
                ),
                *history,
            ],
        )
        confidence = result.confidence
        constraints = result.constraints or constraints
        assumptions = result.assumptions or assumptions
        print(f"  理解：{result.understanding}（{confidence:.0%}）")

        if confidence >= PRE_CONFIDENCE or not result.question:
            break
        print(f"  问：{result.question}")
        answer = ask_user("你（是/否/补充）> ")
        history.append(AIMessage(content=result.question))
        history.append(HumanMessage(content=answer))

    print(f"  约束：{constraints or '无'}")
    print(f"  假设：{assumptions or '无'}")
    return {
        "constraints": constraints,
        "assumptions": assumptions,
        "clarify_history": list(state["clarify_history"]),
    }


# ============ 节点 2：planner ============
def planner(state: PlanClarifyState) -> dict:
    print("\n[Planner] 拆解任务...")
    constraint_text = "\n".join(f"- {c}" for c in state["constraints"]) or "- 无"
    assumption_text = "\n".join(f"- {a}" for a in state["assumptions"]) or "- 无"

    output: PlannerOutput = call_structured(
        PlannerOutput,
        [
            SystemMessage(content="你是任务规划器。基于任务和已确认约束拆出可独立执行的步骤。"),
            HumanMessage(
                content=f"""任务：{state['goal']}

已确认约束：
{constraint_text}

假设：
{assumption_text}

请给出按执行顺序排列的步骤。"""
            ),
        ],
    )

    plan = [{"description": step.description.strip()} for step in output.steps if step.description.strip()]
    for index, step in enumerate(plan, 1):
        print(f"  {index}. {step['description']}")
    return {"plan": plan}


# ============ 节点 3：executor ============
def executor(state: PlanClarifyState) -> dict:
    step = state["plan"][0]
    remaining = state["plan"][1:]
    history = "\n".join(f"- {s['description']}：{s['summary']}" for s in state["past_steps"]) or "- 无"

    print(f"\n[Executor] 执行：{step['description']}")
    try:
        result: StepResult = call_structured(
            StepResult,
            [
                SystemMessage(content="你是执行 Agent。完成指定步骤，输出结论摘要和支撑细节，不要编造数据。"),
                HumanMessage(
                    content=f"""任务：{state['goal']}
当前步骤：{step['description']}
已完成步骤：
{history}

请执行当前步骤。"""
                ),
            ],
        )
        summary, details, ok = result.summary.strip(), result.details.strip(), True
    except Exception as exc:
        summary, details, ok = f"执行失败：{type(exc).__name__}: {exc}", "", False
        print(f"  [降级] 本步失败：{exc}")

    past = list(state["past_steps"])
    past.append(
        {"description": step["description"], "summary": summary, "details": details, "ok": ok}
    )
    print(f"  摘要：{summary[:100]}")
    return {"plan": remaining, "past_steps": past, "executed": state["executed"] + 1}


# ============ 节点 4：replanner ============
def replanner(state: PlanClarifyState) -> dict:
    history = "\n".join(f"- {s['description']}：{s['summary']}" for s in state["past_steps"]) or "- 无"
    remaining = "\n".join(f"- {s['description']}" for s in state["plan"]) or "- 无"
    clarifications = "\n".join(
        f"- 问：{item['question']} / 答：{item['answer']}" for item in state["clarify_history"]
    ) or "- 无"

    print("\n[Replanner] 评估剩余计划...")
    decision: ReplanDecision = call_structured(
        ReplanDecision,
        [
            SystemMessage(
                content=(
                    "你是计划审查员。判断剩余步骤是否仍然必要。"
                    "如果下一步取决于用户的一个选择，action=needs_input，并在 question 里给出一个"
                    "能用是/否回答的问题；不要替用户做假设。"
                )
            ),
            HumanMessage(
                content=f"""任务：{state['goal']}
约束：{'; '.join(state['constraints']) or '无'}

已完成：
{history}

剩余计划：
{remaining}

已有的澄清记录：
{clarifications}

请给出决策：continue / modify / done / needs_input。"""
            ),
        ],
    )

    print(f"  决策：{decision.action}（{decision.reason[:80]}）")
    action = decision.action
    question = decision.question.strip()

    # 澄清预算用尽时，needs_input 降级为 continue，避免卡死
    if action == "needs_input" and (not question or state["clarify_count"] >= MAX_CLARIFY):
        action, question = "continue", ""

    if action == "done":
        return {"plan": [], "replan_action": action}

    if action == "needs_input":
        print(f"  需要用户确认：{question}")
        return {"clarify_question": question, "replan_action": action}

    if action == "modify" and decision.new_steps:
        existing = {s["description"] for s in state["plan"]}
        added = [
            {"description": s.description.strip()}
            for s in decision.new_steps
            if s.description.strip() and s.description.strip() not in existing
        ]
        if added:
            print(f"  追加步骤：{[s['description'] for s in added]}")
            return {"plan": (state["plan"] + added)[:8], "replan_action": action}

    return {"replan_action": action}


# ============ 节点 5：ask_user（按需澄清） ============
def ask_user_node(state: PlanClarifyState) -> dict:
    question = state["clarify_question"]
    print("\n[Clarify] replanner 需要你确认一个选择：")
    print(f"  问：{question}")
    answer = ask_user("你（是/否/补充）> ")

    history = list(state["clarify_history"])
    history.append({"question": question, "answer": answer})
    print()
    return {
        "clarify_history": history,
        "clarify_count": state["clarify_count"] + 1,
        "clarify_question": "",
    }


# ============ 节点 6：finalize ============
def finalize(state: PlanClarifyState) -> dict:
    print("\n[Answer] 生成最终回答...")
    history = "\n".join(
        f"## {s['description']}\n结论：{s['summary']}\n细节：{s['details']}"
        for s in state["past_steps"]
    ) or "（没有已执行的步骤）"
    clarifications = "\n".join(
        f"- {item['question']} → {item['answer']}" for item in state["clarify_history"]
    ) or "- 无"

    answer = call_text(
        [
            SystemMessage(content="你是报告撰写者。基于已执行结果直接回答任务，不要编造未出现的信息。"),
            HumanMessage(
                content=f"""任务：{state['goal']}
约束：{'; '.join(state['constraints']) or '无'}

执行结果：
{history}

澄清记录：
{clarifications}

请生成最终回答。"""
            ),
        ]
    )
    print("\n" + "=" * 60)
    print(answer)
    print("=" * 60 + "\n")
    return {"final_answer": answer}


# ============ 路由 ============
def route_after_replan(state: PlanClarifyState) -> str:
    if state["replan_action"] == "needs_input":
        return "ask_user"
    if not state["plan"] or state["executed"] >= state["max_executions"]:
        return "finalize"
    return "executor"


def build_graph():
    builder = StateGraph(PlanClarifyState)
    builder.add_node("pre_clarify", pre_clarify)
    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.add_node("replanner", replanner)
    builder.add_node("ask_user", ask_user_node)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "pre_clarify")
    builder.add_edge("pre_clarify", "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "replanner")
    builder.add_conditional_edges(
        "replanner",
        route_after_replan,
        {"ask_user": "ask_user", "executor": "executor", "finalize": "finalize"},
    )
    builder.add_edge("ask_user", "replanner")
    builder.add_edge("finalize", END)
    return builder.compile()


# ============ CLI ============
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="计划模式 + 海龟汤澄清")
    parser.add_argument("--goal", help="任务描述；不传则交互输入")
    parser.add_argument("--max-steps", type=int, default=MAX_EXECUTIONS)
    args = parser.parse_args(argv)

    goal = (args.goal or "").strip()
    if not goal:
        try:
            goal = input("任务 > ").strip()
        except (EOFError, KeyboardInterrupt):
            return 1
    if not goal:
        return 2

    print("=" * 60)
    print("计划模式 + 海龟汤澄清")
    print("=" * 60)
    print(f"任务：{goal}\n")

    graph = build_graph()
    initial: PlanClarifyState = {
        "goal": goal,
        "constraints": [],
        "assumptions": [],
        "clarify_history": [],
        "clarify_count": 0,
        "clarify_question": "",
        "plan": [],
        "past_steps": [],
        "final_answer": "",
        "executed": 0,
        "max_executions": max(1, args.max_steps),
        "replan_action": "",
    }

    try:
        final = graph.invoke(initial, config={"recursion_limit": 30})
    except UserExit:
        print("\n已退出（未生成最终回答）")
        return 0
    except Exception as exc:
        print(f"\n[错误] {type(exc).__name__}: {exc}")
        return 1

    print(f"执行步数：{final['executed']}；按需澄清次数：{final['clarify_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
