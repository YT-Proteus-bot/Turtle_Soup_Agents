# agents_learn_demo/observation_reflect.py
# ==============================================================================
# Reflection 版观察闸门：不设独立的 check 节点，让 observation 节点“问自己”。
#
# 与 observation_gate.py 的区别：
#   gate：tools → check_observation（独立节点）→ 判定 sufficient / confidence
#   reflect：tools → observe（同一节点内两段式）：
#              1) 先写观察草稿；
#              2) 再用海龟汤思路对自己提二元检查题（是/否/unknown + 证据）；
#              3) 由代码按权重算出把握度，而不是让模型直接报一个百分数。
#
# 这样做的目的：把“把握度”从主观打分变成可审计的清单得分。
#
# 图结构：
#   START → plan_action → tools → observe ─┬─ 把握度 >= 95% → answer → END
#                                          ├─ blocked        → answer → END
#                                          └─ 不足           → plan_action
#
# 运行：
#   .\.venv\Scripts\python.exe agents_learn_demo\observation_reflect.py
#   .\.venv\Scripts\python.exe agents_learn_demo\observation_reflect.py "checkout 为什么变慢了？"
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

MAX_ROUNDS = 5
CONFIDENCE_THRESHOLD = 0.95
MIN_CHECKS = 3          # 少于 3 条检查项时，把握度封顶 0.8
NO_CRITICAL_CAP = 0.85  # 一条 critical 检查都没有时，把握度封顶 0.85

# ============ 模拟环境 ============
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
        "db": "连接池 20/20 已用满，14 个请求排队，最长等待 3.2s；DB 侧慢查询 3 条，单条最慢 4.1s",
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
        "db": "连接池 5/20，无排队，慢查询 0 条",
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


@tool
def get_db_pool(service: str) -> str:
    """查询服务对应的数据库连接池使用情况：占用、排队、慢查询。"""
    data = SERVICE_DATA.get(service)
    if not data:
        return f"未知服务：{service}"
    return f"{service} 的 DB 连接池：{data['db']}"


TOOL_MAP = {
    "status": get_service_status,
    "metrics": get_metrics,
    "errors": get_recent_errors,
    "db_pool": get_db_pool,
}


def format_observations(observations: list[dict]) -> str:
    if not observations:
        return "（还没有任何观察）"
    return "\n".join(
        f"{index}. [{item['tool']}] {item['result']}"
        for index, item in enumerate(observations, 1)
    )


# ============ 结构化输出模型 ============
class NextAction(BaseModel):
    tool: Literal["status", "metrics", "errors", "db_pool", "none"] = Field(
        description="要调用的工具；none 表示没工具可查"
    )
    service: str = Field(default="checkout")
    metric: Optional[Literal["latency", "cpu", "memory", "error_rate"]] = None
    minutes: int = 30
    reason: str = ""


class SelfCheck(BaseModel):
    """一条自我检查项。"""

    question: str = Field(min_length=3, max_length=200, description="能用是/否回答的检查问题")
    answer: Literal["yes", "no", "unknown"] = Field(description="自检结论；没有证据时必须 unknown")
    evidence: str = Field(default="", description="支持答案的观察原文；answer=yes 时必填")
    weight: int = Field(default=1, ge=1, le=3, description="该检查项的重要性权重 1-3")
    critical: bool = Field(default=False, description="是否关键项：不满足时把握度直接封顶 0.9")


class ObservationReview(BaseModel):
    """observation 节点自检的结果。"""

    checks: list[SelfCheck] = Field(min_length=1, max_length=8, description="3-6 条二元检查项")
    critical_missing: list[str] = Field(default_factory=list, description="关键但未满足的点")


class ObservationSummary(BaseModel):
    answer: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)


# ============ 图状态 ============
class ReflectState(TypedDict):
    question: str
    observations: list[dict]
    drafts: list[str]
    reviews: list[dict]
    pending: dict
    executed: list[str]
    rounds: int
    status: str
    answer: str


_planner = llm.with_structured_output(NextAction, method="function_calling")
_reviewer = llm.with_structured_output(ObservationReview, method="function_calling")
_answerer = llm.with_structured_output(ObservationSummary, method="function_calling")


# ============ 节点 1：plan_action ============
def plan_action(state: ReflectState) -> dict:
    last_review = state["reviews"][-1] if state["reviews"] else None
    if last_review:
        feedback = (
            f"上一轮自检把握度 {last_review['confidence']:.0%}，未满足项："
            f"{'；'.join(last_review['critical_missing']) or '见检查清单'}。"
            f"下一轮请补齐缺口。"
        )
    else:
        feedback = "还没有自检结果，先做一个基础观察（服务状态）。"

    action: NextAction = _planner.invoke(
        [
            SystemMessage(
                content=(
                    "你是排障 Agent 的规划器。每轮只能选择一个工具调用，不能直接回答；"
                    "是否足够由观察自检决定。"
                    "可用工具：status（服务状态）、metrics（指标）、errors（错误日志）、"
                    "db_pool（数据库连接池）、none（不再查询）。"
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
def run_tool(state: ReflectState) -> dict:
    action = state["pending"]
    tool_name = action["tool"]

    if tool_name == "none":
        print("[Tools] 模型没有选择工具，交给观察节点自检现有内容。")
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

    try:
        result = TOOL_MAP[tool_name].invoke(args)
    except Exception as exc:
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


# ============ 节点 3：observe（草稿 + 自检） ============
def make_draft(question: str, observations: list[dict]) -> str:
    """第一段：对当前观察做一个不带评分的解读草稿。"""
    resp = llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是排障分析员。根据观察写一段分析草稿，严格基于观察原文，"
                    "不要推测观察之外的时间线、因果或外部信息；"
                    "明确区分“已观察到的事实”和“仍然未知的部分”。"
                )
            ),
            HumanMessage(
                content=f"""用户问题：{question}

观察：
{format_observations(observations)}

请写一段分析草稿：这些观察说明了什么、还缺什么。"""
            ),
        ]
    )
    content = resp.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def review_observation(question: str, observations: list[dict], draft: str) -> ObservationReview:
    """第二段：用海龟汤思路对自己提二元检查题，并给证据。"""
    return _reviewer.invoke(
        [
            SystemMessage(
                content=(
                    "你是自己的审查员。针对“这些观察是否足以回答用户问题”提出 3-6 条二元检查项，"
                    "每条回答 yes/no/unknown，并引用观察原文作为 evidence。"
                    "没有原文证据的必须回答 unknown。"
                    "请输出恰好 5 条检查项，分别对应下面五类，不要增加其他检查项："
                    "1) 是否确认了对象状态；2) 是否有支撑结论的关键数据；"
                    "3) 是否定位到原因/瓶颈；4) 是否直接回答了用户问题；"
                    "5) 结论是否都有观察原文证据。"
                    "每项回答 yes/no/unknown，并在 evidence 里引用观察原文。"
                    "没有原文证据的必须回答 unknown。"
                    "第 3 项和第 4 项必须标记 critical=true；其余项也可以按重要性标 critical。"
                    "只提出当前观察能够直接回答的检查项；需要外部对比、历史基线、"
                    "因果证明这类观察里没有的信息，不要作为检查项。"
                )
            ),
            HumanMessage(
                content=f"""用户问题：{question}

观察：
{format_observations(observations)}

草稿：
{draft}

请输出检查清单。"""
            ),
        ]
    )


def compute_confidence(checks: list[SelfCheck]) -> float:
    """按权重把 yes/no/unknown 聚合成把握度；critical 未满足时封顶 0.9。"""
    total = sum(check.weight for check in checks)
    if total <= 0:
        return 0.0
    yes = sum(check.weight for check in checks if check.answer == "yes")
    confidence = yes / total

    if any(check.critical and check.answer != "yes" for check in checks):
        confidence = min(confidence, 0.9)
    if not any(check.critical for check in checks):
        confidence = min(confidence, NO_CRITICAL_CAP)
    if len(checks) < MIN_CHECKS:
        confidence = min(confidence, 0.8)
    return round(confidence, 3)


def observe(state: ReflectState) -> dict:
    """观察节点：先解读，再自检，最后由代码算把握度。"""
    question = state["question"]
    observations = state["observations"]

    draft = make_draft(question, observations)
    review = review_observation(question, observations, draft)

    # 空证据的“yes”一律降级为 unknown，防止模型空口断言
    for check in review.checks:
        if check.answer == "yes" and not check.evidence.strip():
            check.answer = "unknown"

    confidence = compute_confidence(review.checks)

    print("\n[Reflect] 草稿：")
    print("  " + draft.replace("\n", "\n  ")[:400])
    print("[Reflect] 自检：")
    for check in review.checks:
        mark = {"yes": "[是]", "no": "[否]", "unknown": "[?]"}[check.answer]
        critical = " *critical" if check.critical else ""
        print(f"  {mark} {check.question} (w={check.weight}{critical})")
        if check.evidence:
            print(f"       证据：{check.evidence[:100]}")
    print(f"[Reflect] 代码计算把握度：{confidence:.0%}")

    reviews = list(state["reviews"])
    reviews.append(
        {
            "confidence": confidence,
            "checks": [check.model_dump() for check in review.checks],
            "critical_missing": review.critical_missing,
        }
    )
    drafts = list(state["drafts"]) + [draft]

    if confidence >= CONFIDENCE_THRESHOLD:
        print("[Reflect] 达到阈值，可以回答。")
        return {"reviews": reviews, "drafts": drafts, "status": "done"}
    if state["status"] == "no_tool":
        print("[Reflect] 模型已无工具可查，按现有观察收尾。")
        return {"reviews": reviews, "drafts": drafts, "status": "blocked"}
    if state["rounds"] >= MAX_ROUNDS:
        print(f"[Reflect] 已达到最大查询轮数 {MAX_ROUNDS}，按现有观察收尾。")
        return {"reviews": reviews, "drafts": drafts, "status": "blocked"}
    return {"reviews": reviews, "drafts": drafts, "status": "continue"}


# ============ 节点 4：answer ============
def answer(state: ReflectState) -> dict:
    last_review = state["reviews"][-1] if state["reviews"] else None
    draft = state["drafts"][-1] if state["drafts"] else ""
    gap_note = ""
    if state["status"] == "blocked" and last_review:
        missing = last_review.get("critical_missing") or [
            check["question"] for check in last_review["checks"] if check["answer"] != "yes"
        ]
        gap_note = f"\n注意：当前观察并不充分，未满足项：{'; '.join(missing)[:200]}。请在回答中说明这一缺口。"

    summary: ObservationSummary = _answerer.invoke(
        [
            SystemMessage(
                content=(
                    "你是排障助手。只根据给定的观察回答，不要编造数据。"
                    "先回答“有没有问题”，再说明“瓶颈/原因”。"
                )
            ),
            HumanMessage(
                content=f"""用户问题：{state['question']}

观察：
{format_observations(state['observations'])}

分析草稿：
{draft}
{gap_note}

请给出最终回答，并列出关键证据。"""
            ),
        ]
    )

    print("\n" + "=" * 60)
    print(summary.answer)
    print("=" * 60 + "\n")
    return {"answer": summary.answer}


# ============ 路由与组图 ============
def route_after_tools(state: ReflectState) -> str:
    if state["status"] == "blocked":
        return "answer"
    return "observe"


def route_after_observe(state: ReflectState) -> str:
    if state["status"] in ("done", "blocked"):
        return "answer"
    return "plan_action"


def build_graph():
    builder = StateGraph(ReflectState)
    builder.add_node("plan_action", plan_action)
    builder.add_node("tools", run_tool)
    builder.add_node("observe", observe)
    builder.add_node("answer", answer)

    builder.add_edge(START, "plan_action")
    builder.add_edge("plan_action", "tools")
    builder.add_conditional_edges(
        "tools", route_after_tools, {"observe": "observe", "answer": "answer"}
    )
    builder.add_conditional_edges(
        "observe", route_after_observe, {"plan_action": "plan_action", "answer": "answer"}
    )
    builder.add_edge("answer", END)
    return builder.compile()


DEFAULT_QUESTION = "checkout 服务现在有没有问题？如果有，瓶颈可能在哪？"


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    question = " ".join(argv).strip() or DEFAULT_QUESTION

    print("=" * 60)
    print("Reflection 版观察闸门")
    print("=" * 60)
    print(f"问题：{question}\n")

    graph = build_graph()
    initial: ReflectState = {
        "question": question,
        "observations": [],
        "drafts": [],
        "reviews": [],
        "pending": {},
        "executed": [],
        "rounds": 0,
        "status": "continue",
        "answer": "",
    }
    final = graph.invoke(initial, config={"recursion_limit": 20})

    print("查询轮数：", final["rounds"])
    if final["reviews"]:
        last = final["reviews"][-1]
        print(f"最终把握度（代码计算）：{last['confidence']:.0%}")
        print("自检明细：")
        for check in last["checks"]:
            print(f"  [{check['answer']}] {check['question']} (w={check['weight']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
