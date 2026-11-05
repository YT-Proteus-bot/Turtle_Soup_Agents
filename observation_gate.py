# agents_learn_demo/observation_gate.py
# ==============================================================================
# 观察充分性闸门（Observation Gate）Agent demo。
#
# 和 intent_clarifier / observation_clarifier 的区别：
#   那两个是“向用户提问”，这个 demo 是“向环境提问”——
#   每轮工具执行完，check_observation 节点判断：
#       本轮 output 是否足够回答用户需求？
#       足够 → 回答；不够 → 指出缺什么，并让模型带着更窄的查询再查一次。
#
# 图结构：
#   START → plan_action → tools → check_observation ─┬─ sufficient → answer → END
#                                                    ├─ blocked    → answer → END
#                                                    └─ insufficient → plan_action（循环）
#
# 运行：
#   .\.venv\Scripts\python.exe agents_learn_demo\observation_gate.py
#   .\.venv\Scripts\python.exe agents_learn_demo\observation_gate.py "checkout 为什么变慢了？"
# ==============================================================================

from __future__ import annotations

import os
import sys
from typing import Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
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

MAX_ROUNDS = 5                # 最多查几轮，防止“查了还不够”死循环
CONFIDENCE_THRESHOLD = 0.95   # 观察充分性的把握度门槛

# ============ 模拟环境 ============
# 这是“世界状态”，模型只能通过工具看到，不能直接读。
SERVICE_DATA: dict[str, dict[str, Any]] = {
    "checkout": {
        "status": "running，version 2.3.1，部署于 10:04（约 8 分钟前）",
        "metrics": {
            "latency": "P95 850ms（基线 200ms），最近 30 分钟持续升高",
            "cpu": "平均 92%，峰值 98%（最近 30 分钟）",
            "memory": "平均 71%，无 swap",
            "error_rate": "5.4%（基线 0.2%）",
        },
        "errors": "最近 30 分钟 37 条 ERROR：DB connection timeout 29 条、connection pool exhausted 5 条、其他 3 条",
    },
    "payments": {
        "status": "running，version 4.1.0，部署于昨天 22:10",
        "metrics": {
            "latency": "P95 210ms（基线 200ms）",
            "cpu": "平均 43%",
            "memory": "平均 62%",
            "error_rate": "0.3%（基线 0.2%）",
        },
        "errors": "最近 30 分钟 2 条 WARN，无 ERROR",
    },
}


@tool
def get_service_status(service: str) -> str:
    """查询服务当前状态：是否在运行、版本、部署时间。"""
    data = SERVICE_DATA.get(service)
    if not data:
        return f"未知服务：{service}"
    return f"{service}：{data['status']}"


@tool
def get_metrics(service: str, metric: str, minutes: int = 30) -> str:
    """查询服务指标（latency/cpu/memory/error_rate）最近 N 分钟的情况。"""
    data = SERVICE_DATA.get(service)
    if not data:
        return f"未知服务：{service}"
    value = data["metrics"].get(metric)
    if value is None:
        return f"{service} 没有指标 {metric}；可用指标：{', '.join(data['metrics'])}"
    return f"{service} 的 {metric}（最近 {minutes} 分钟）：{value}"


@tool
def get_recent_errors(service: str, minutes: int = 30) -> str:
    """查询服务最近 N 分钟的 ERROR 日志数量与主要类型。"""
    data = SERVICE_DATA.get(service)
    if not data:
        return f"未知服务：{service}"
    return f"{service}：{data['errors']}"


TOOL_MAP = {"status": get_service_status, "metrics": get_metrics, "errors": get_recent_errors}


def format_observations(observations: list[dict]) -> str:
    """把已收集的观察渲染成文本。"""
    if not observations:
        return "（还没有任何观察）"
    lines = []
    for index, item in enumerate(observations, 1):
        lines.append(f"{index}. [{item['tool']}] {item['result']}")
    return "\n".join(lines)


# ============ 结构化输出模型 ============
class NextAction(BaseModel):
    """plan_action 节点：这一轮该调哪个工具。"""

    tool: Literal["status", "metrics", "errors", "none"] = Field(
        description="要调用的工具；none 表示模型认为没工具可查了"
    )
    service: str = Field(default="checkout", description="目标服务名")
    metric: Optional[Literal["latency", "cpu", "memory", "error_rate"]] = Field(
        default=None, description="tool=metrics 时指定指标"
    )
    minutes: int = Field(default=30, description="查询最近多少分钟")
    reason: str = Field(default="", description="为什么查这个")


class ObservationCheck(BaseModel):
    """check_observation 节点：本轮观察是否足够回答用户需求。"""

    sufficient: bool = Field(description="当前观察是否足够回答用户的问题")
    confidence: float = Field(ge=0.0, le=1.0, description="对“观察是否足够”的把握度，按信息完整度给分 0-1")
    missing: list[str] = Field(default_factory=list, description="还缺什么信息，最多 3 条")
    next_focus: str = Field(default="", description="如果不够，下一轮应该查什么（具体一点）")
    should_ask_user: bool = Field(default=False, description="工具解决不了、需要问用户时为 True")


class ObservationSummary(BaseModel):
    """最终回答。"""

    answer: str = Field(min_length=1, description="给用户的最终回答")
    evidence: list[str] = Field(default_factory=list, description="支撑结论的关键观察")


# ============ 图状态 ============
class GateState(TypedDict):
    question: str
    observations: list[dict]     # [{"tool": ..., "args": ..., "result": ...}]
    checks: list[dict]           # 每轮 check 的结果
    pending: dict                # 本轮待执行的工具调用
    executed: list[str]          # 已执行过的 (tool, args) 签名，用于去重
    rounds: int
    status: str                  # continue / done / blocked
    answer: str


# ============ 结构化输出模型与三个 LLM 调用 ============
_planner = llm.with_structured_output(NextAction, method="function_calling")
_checker = llm.with_structured_output(ObservationCheck, method="function_calling")
_answerer = llm.with_structured_output(ObservationSummary, method="function_calling")


# ============ 节点 1：plan_action ============
def plan_action(state: GateState) -> dict:
    """根据问题、已有观察、上一轮检查结果，决定本轮查什么。"""
    last_check = state["checks"][-1] if state["checks"] else None
    feedback = (
        f"上一轮检查：不够充分。缺失：{'; '.join(last_check['missing']) or '未说明'}；"
        f"下一轮重点：{last_check['next_focus'] or '未说明'}"
        if last_check and not last_check["sufficient"]
        else "还没有检查结果，先做一个基础观察（服务状态）。"
    )

    action: NextAction = _planner.invoke(
        [
            SystemMessage(
                content=(
                    "你是排障 Agent 的规划器。你每一轮只能选择一个工具调用，不能直接回答；"
                    "是否足够回答由检查节点决定。优先查最基础、信息量最大的数据。"
                )
            ),
            HumanMessage(
                content=f"""用户问题：{state['question']}

已收集的观察：
{format_observations(state['observations'])}

{feedback}

请决定这一轮调用哪个工具、参数是什么。"""
            ),
        ]
    )

    print(f"\n[Plan] 决定查：{action.tool} "
          f"(service={action.service}, metric={action.metric}, minutes={action.minutes})")
    return {"pending": action.model_dump()}


# ============ 节点 2：tools ============
def run_tool(state: GateState) -> dict:
    """执行 plan_action 选定的工具，把结果作为 observation 追加。"""
    action = state["pending"]
    tool_name = action["tool"]

    if tool_name == "none":
        # 模型明确表示没有可查的工具：不新增观察，交给 check 判定现有观察够不够
        print("[Tools] 模型没有选择工具，交给检查节点判断现有观察。")
        return {"status": "no_tool"}

    args: dict[str, Any] = {"service": action["service"]}
    if tool_name == "metrics":
        args["metric"] = action.get("metric") or "latency"
        args["minutes"] = action.get("minutes", 30)
    elif tool_name == "errors":
        args["minutes"] = action.get("minutes", 30)

    signature = f"{tool_name}:{sorted(args.items())}"
    if signature in state["executed"]:
        print(f"[Tools] 重复查询（{signature}），停止。")
        return {"status": "blocked"}

    tool_fn = TOOL_MAP[tool_name]
    try:
        result = tool_fn.invoke(args)
    except Exception as exc:  # 工具参数不合法等
        result = f"工具执行失败：{type(exc).__name__}: {exc}"

    print(f"[Tools] {tool_name}({args}) -> {result[:120]}")
    observations = list(state["observations"])
    observations.append({"tool": tool_name, "args": args, "result": result})
    return {
        "observations": observations,
        "executed": list(state["executed"]) + [signature],
        "rounds": state["rounds"] + 1,
        "status": "continue",
    }


# ============ 节点 3：check_observation ============
def check_observation(state: GateState) -> dict:
    """判断本轮观察是否足够回答用户需求；不够就给出下一轮重点。"""
    check: ObservationCheck = _checker.invoke(
        [
            SystemMessage(
                content=(
                    "你是观察质检员。判断当前工具输出是否足以回答用户的问题。"
                    "必须能同时回答两件事才算足够：(1) 服务/对象有没有问题；"
                    "(2) 问题出在哪里/瓶颈是什么。只看到状态、没看到指标或错误，不算足够。"
                    "confidence 按信息完整度给分：只有状态 <0.4；状态+一个指标 <0.7；"
                    "指标+错误 <0.9；能定位瓶颈 >=0.95。不要虚高；如果 sufficient=false，"
                    "confidence 必须低于 0.9。"
                )
            ),
            HumanMessage(
                content=f"""用户问题：{state['question']}

已收集的观察：
{format_observations(state['observations'])}

请判断：这些观察是否足够回答用户问题？如果不够，说明缺什么、下一轮该查什么。"""
            ),
        ]
    )

    sufficient = check.sufficient and check.confidence >= CONFIDENCE_THRESHOLD
    print(f"[Check] sufficient={check.sufficient}, confidence={check.confidence:.0%}, "
          f"missing={check.missing}")

    checks = list(state["checks"])
    checks.append(
        {
            "sufficient": sufficient,
            "confidence": check.confidence,
            "missing": check.missing,
            "next_focus": check.next_focus,
            "should_ask_user": check.should_ask_user,
        }
    )

    if sufficient:
        return {"checks": checks, "status": "done"}
    if check.should_ask_user:
        print("[Check] 工具无法解决，需要用户补充信息。")
        return {"checks": checks, "status": "blocked"}
    if state["status"] == "no_tool":
        print("[Check] 模型已无工具可查，按现有观察收尾。")
        return {"checks": checks, "status": "blocked"}
    if state["rounds"] >= MAX_ROUNDS:
        print(f"[Check] 已达到最大查询轮数 {MAX_ROUNDS}，按现有观察收尾。")
        return {"checks": checks, "status": "blocked"}
    return {"checks": checks, "status": "continue"}


# ============ 节点 4：answer ============
def answer(state: GateState) -> dict:
    """基于已收集的观察写最终回答；不充分时明确说明缺口。"""
    last_check = state["checks"][-1] if state["checks"] else None
    gap_note = ""
    if state["status"] == "blocked" and last_check:
        gap_note = f"\n注意：当前观察并不充分，缺失信息：{'; '.join(last_check['missing']) or '未说明'}。请在回答中明确说明这一点。"

    summary: ObservationSummary = _answerer.invoke(
        [
            SystemMessage(
                content=(
                    "你是排障助手。只根据给定的观察回答用户问题，不要编造工具里没有的数据。"
                    "先回答“有没有问题”，再说明“瓶颈/原因”。"
                )
            ),
            HumanMessage(
                content=f"""用户问题：{state['question']}

已收集的观察：
{format_observations(state['observations'])}
{gap_note}

请给出最终回答，并列出关键证据。"""
            ),
        ]
    )

    print("\n" + "=" * 60)
    print(summary.answer)
    print("=" * 60 + "\n")
    return {"answer": summary.answer}


# ============ 路由 ============
def route_after_tools(state: GateState) -> str:
    """工具层直接判定无法继续（重复查询）时跳过检查，直接收尾。"""
    if state["status"] == "blocked":
        return "answer"
    return "check_observation"


def route_after_check(state: GateState) -> str:
    if state["status"] in ("done", "blocked"):
        return "answer"
    return "plan_action"


# ============ 组图 ============
def build_graph():
    builder = StateGraph(GateState)
    builder.add_node("plan_action", plan_action)
    builder.add_node("tools", run_tool)
    builder.add_node("check_observation", check_observation)
    builder.add_node("answer", answer)

    builder.add_edge(START, "plan_action")
    builder.add_edge("plan_action", "tools")
    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {"check_observation": "check_observation", "answer": "answer"},
    )
    builder.add_conditional_edges(
        "check_observation",
        route_after_check,
        {"plan_action": "plan_action", "answer": "answer"},
    )
    builder.add_edge("answer", END)
    return builder.compile()


# ============ 运行 ============
DEFAULT_QUESTION = "checkout 服务现在有没有问题？如果有，瓶颈可能在哪？"


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    question = " ".join(argv).strip() or DEFAULT_QUESTION

    print("=" * 60)
    print("观察充分性闸门（Observation Gate）")
    print("=" * 60)
    print(f"问题：{question}\n")

    graph = build_graph()
    initial: GateState = {
        "question": question,
        "observations": [],
        "checks": [],
        "pending": {},
        "executed": [],
        "rounds": 0,
        "status": "continue",
        "answer": "",
    }
    final = graph.invoke(initial, config={"recursion_limit": 20})

    print("查询轮数：", final["rounds"])
    print("收集到的观察：")
    for item in final["observations"]:
        print(f"  - [{item['tool']}] {item['result'][:100]}")
    confidence = final["checks"][-1]["confidence"] if final["checks"] else 0.0
    print(f"最终把握度：{confidence:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
