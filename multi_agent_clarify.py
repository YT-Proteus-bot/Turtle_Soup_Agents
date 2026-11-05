# agents_learn_demo/multi_agent_clarify.py
# ==============================================================================
# 多 Agent 海龟汤协作 demo（最小版）。
#
# 角色：
#   Organizer（组织者）：拆活、派活、回答下属的确认问题、答不上来时升级给用户。
#   Retrieval 员工：负责检索资料。
#   Writing 员工：负责把资料写成结论。
#   Reviewer：最后用二元检查验收。
#
# 演示的原则：
#   1. 派活前先确认能力（这里查能力清单，不浪费一次模型调用）；
#   2. 员工有歧义时向组织者提一个“是/否”问题，而不是自己猜；
#   3. 组织者只根据任务和已确认约束回答，答不上来就 unknown 并升级给用户；
#   4. 审查者用二元检查验收，不通过让 Writing 带着缺口重写一次。
#
# 运行：
#   .\.venv\Scripts\python.exe agents_learn_demo\multi_agent_clarify.py
# ==============================================================================

from __future__ import annotations

import os
from typing import Callable, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
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

RETRY_ATTEMPTS = 3

# ============ 员工能力清单 ============
# 派活前先查清单，而不是问模型“你能做吗”——agent 之间也不该问能确定性查到的事。
CAPABILITIES = {
    "retrieval": "工具：search，可检索本地资料库",
    "writing": "能力：中文写作，200 字以内，会引用资料编号",
}

# ============ 本地资料 ============
DOCS = [
    {
        "id": "d1",
        "title": "LangGraph 简介",
        "content": "LangGraph 是 LangChain 团队推出的低层 Agent 编排框架，用状态图（StateGraph）"
                   "组织节点和边，支持循环、条件分支、检查点持久化和人工介入。",
    },
    {
        "id": "d2",
        "title": "CrewAI 简介",
        "content": "CrewAI 是面向多 Agent 协作的高层框架，以 Role、Task、Crew 为核心概念，"
                   "强调角色分工和任务委派，适合快速搭建流程明确的多 Agent 应用。",
    },
    {
        "id": "d3",
        "title": "AutoGen 简介",
        "content": "AutoGen 是微软推出的多 Agent 对话框架，多个 Agent 通过消息对话协作完成任务。",
    },
    {
        "id": "d4",
        "title": "框架选型建议",
        "content": "需要精细控制状态和流程时选 LangGraph；需要快速搭建角色分工的协作流程时选 CrewAI。",
    },
]


# ============ 结构化输出模型 ============
class ScopeQuestion(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0, description="对任务边界的把握度")
    question: Optional[str] = Field(default=None, description="需要向用户确认的一个是/否问题")
    constraints: list[str] = Field(default_factory=list, description="已确认的约束")


class WorkerQuestion(BaseModel):
    question: Optional[str] = Field(default=None, description="员工需要向组织者确认的是/否问题")
    reason: str = Field(default="", max_length=200, description="为什么需要确认")


class OrganizerAnswer(BaseModel):
    answer: Literal["yes", "no", "unknown"] = Field(description="能确定就回答，不能确定就 unknown")
    evidence: str = Field(default="", max_length=300, description="回答依据；unknown 时留空")
    escalate: bool = Field(default=False, description="是否需要升级给用户")


class ReviewCheck(BaseModel):
    question: str = Field(description="一条能用是/否判断的验收项")
    passed: bool = Field(description="是否通过")
    evidence: str = Field(default="", max_length=300, description="依据")


class Review(BaseModel):
    checks: list[ReviewCheck] = Field(min_length=1, max_length=6)
    passed: bool = Field(description="是否全部通过")
    missing: list[str] = Field(default_factory=list, description="没通过的点")


# ============ 用户输入 ============
ASK: Callable[[str], str] = input
EXIT_WORDS = {"/exit", "exit", "quit", "q"}


class UserExit(Exception):
    pass


def ask_user(prompt: str) -> str:
    raw = ASK(prompt).strip()
    if raw.lower() in EXIT_WORDS:
        raise UserExit
    return raw


# ============ LLM 调用 ============
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
            print(f"  [重试] {schema.__name__} 第 {attempt}/{RETRY_ATTEMPTS} 次：{type(exc).__name__}: {exc}")
    raise RuntimeError(f"{schema.__name__} 连续 {RETRY_ATTEMPTS} 次失败") from last_error


def call_text(messages: list) -> str:
    resp = llm.invoke(messages)
    content = resp.content
    return content.strip() if isinstance(content, str) else str(content).strip()


# ============ 派活前：能力确认（查清单，不调模型） ============
def capability_check(role: str) -> None:
    print(f"[Organizer → {role}] 这个任务需要你的能力，你这边可以吗？")
    print(f"[{role} → Organizer] 是。依据：{CAPABILITIES[role]}")


# ============ 组织者 → 用户：范围确认 ============
def ask_scope(task: str) -> list[str]:
    print("\n[Organizer] 先确认任务边界...")
    result: ScopeQuestion = call_structured(
        ScopeQuestion,
        [
            SystemMessage(
                content=(
                    "你是组织者。任务如果缺少关键范围（长度、受众、是否要引用），"
                    "向用户提一个是/否问题确认；范围足够明确时 question 留空。"
                    "constraints 里写已经能确定的约束。"
                )
            ),
            HumanMessage(content=f"任务：{task}"),
        ],
    )
    constraints = list(result.constraints)
    print(f"  把握度：{result.confidence:.0%}；已确认约束：{constraints or '无'}")

    if result.question and result.confidence < 0.9:
        print(f"[Organizer → User] {result.question}")
        answer = ask_user("你（是/否/补充）> ")
        constraints.append(f"{result.question} → {answer}")
        print(f"[User → Organizer] {answer}")
    return constraints


# ============ 员工 → 组织者：一个确认问题 ============
def worker_asks(role: str, task: str, constraints: list[str]) -> Optional[str]:
    result: WorkerQuestion = call_structured(
        WorkerQuestion,
        [
            SystemMessage(
                content=(
                    f"你是 {role} 员工。开始执行前，如果任务或约束里有影响你工作的是非歧义，"
                    "向组织者提一个是/否问题；没有就留空。不要问能自己查到的信息。"
                )
            ),
            HumanMessage(
                content=f"任务：{task}\n已确认约束：{constraints or '无'}"
            ),
        ],
    )
    if not result.question:
        print(f"[{role}] 没有需要确认的问题，开始执行。")
        return None
    print(f"[{role} → Organizer] {result.question}（原因：{result.reason}）")
    return result.question


# ============ 组织者 → 员工：回答或升级 ============
def organizer_answers(role: str, question: str, task: str, constraints: list[str]) -> OrganizerAnswer:
    answer: OrganizerAnswer = call_structured(
        OrganizerAnswer,
        [
            SystemMessage(
                content=(
                    "你是组织者。只能根据任务和已确认约束回答下属的问题。"
                    "能确定就回答 yes/no 并给出依据；不能确定就 answer=unknown、escalate=true，不要猜。"
                )
            ),
            HumanMessage(
                content=f"""任务：{task}
已确认约束：{constraints or '无'}
{role} 的问题：{question}"""
            ),
        ],
    )
    print(f"[Organizer → {role}] {answer.answer}（{answer.evidence or '无依据'}）")
    return answer


def resolve_worker_question(role: str, question: str, task: str, constraints: list[str]) -> str:
    """回答员工问题；返回最终答案文本（供去重复用）。"""
    answer = organizer_answers(role, question, task, constraints)
    if answer.answer == "unknown" or answer.escalate:
        print("[Organizer → User] 这个问题我无法确定，需要你确认：")
        user_answer = ask_user(f"关于「{question}」，你（是/否/补充）> ")
        constraints.append(f"{question} → 用户回答：{user_answer}")
        print(f"[User → Organizer] {user_answer}")
        return user_answer
    constraints.append(f"{question} → {answer.answer}（{answer.evidence}）")
    return f"{answer.answer}（{answer.evidence}）"


# ============ 检索员工 ============
def _normalize(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _bigrams(text: str) -> set[str]:
    return {text[i:i + 2] for i in range(len(text) - 1)}


def questions_similar(left: str, right: str, threshold: float = 0.5) -> bool:
    """判断两个问题是否高度重复，用于多 Agent 之间的提问去重。"""
    ga, gb = _bigrams(_normalize(left)), _bigrams(_normalize(right))
    if not ga or not gb:
        return False
    return len(ga & gb) / len(ga | gb) >= threshold


def retrieve(query: str, top_k: int = 2) -> list[dict]:
    print("\n[Retrieval] 检索资料...")
    query_norm = _normalize(query)
    qchars = set(query_norm)
    qgrams = _bigrams(query_norm)
    scored = []
    for doc in DOCS:
        doc_norm = _normalize(doc["title"] + doc["content"])
        chars = qchars & set(doc_norm)
        uni = len(chars) / len(qchars) if qchars else 0.0
        grams = qgrams & _bigrams(doc_norm)
        bi = len(grams) / len(qgrams) if qgrams else 0.0
        scored.append((0.5 * uni + 0.5 * bi, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    hits = [doc for score, doc in scored[:top_k] if score > 0]
    for doc in hits:
        print(f"  命中 [{doc['id']}] {doc['title']}")
    return hits


# ============ 写作员工 ============
def write(task: str, snippets: list[dict], constraints: list[str], feedback: str = "") -> str:
    print("\n[Writing] 编写结论...")
    material = "\n".join(f"[{doc['id']}] {doc['content']}" for doc in snippets) or "（无资料）"
    prompt = f"""任务：{task}
已确认约束：
{chr(10).join(f'- {c}' for c in constraints) or '- 无'}

资料：
{material}

请写一段不超过 200 字的结论，必须引用资料编号，例如 [d1]。"""
    if feedback:
        prompt += f"\n上一版的问题：{feedback}。请修正。"
    text = call_text([SystemMessage(content="你是写作员工。只根据资料写，不要编造。"), HumanMessage(content=prompt)])
    print(f"  草稿：{text[:80]}...")
    return text


# ============ 审查者 ============
def review(task: str, draft: str) -> Review:
    print("\n[Reviewer] 验收...")
    result: Review = call_structured(
        Review,
        [
            SystemMessage(
                content=(
                    "你是审查者。用 3 条二元检查验收：1) 是否同时覆盖任务要求的对象；"
                    "2) 是否有明确结论；3) 是否引用了资料编号。"
                    "每条给出 passed 和 evidence，最后给整体 passed 与 missing。"
                )
            ),
            HumanMessage(content=f"任务：{task}\n草稿：{draft}"),
        ],
    )
    for check in result.checks:
        print(f"  [{'通过' if check.passed else '未过'}] {check.question}")
    return result


# ============ 主流程 ============
def run(task: str) -> dict:
    print("=" * 60)
    print("多 Agent 海龟汤协作")
    print("=" * 60)
    print(f"任务：{task}\n")

    constraints = ask_scope(task)
    resolved: list[dict] = []

    # 派活前确认能力 + 收集员工问题（重复的问题直接复用已确认结论）
    for role in ("retrieval", "writing"):
        print()
        capability_check(role)
        question = worker_asks(role, task, constraints)
        if not question:
            continue

        matched = next(
            (item for item in resolved if questions_similar(question, item["question"])), None
        )
        if matched:
            print(f"[Organizer] 这个问题之前已确认，直接复用：{matched['answer']}")
            constraints.append(f"{question} → 复用：{matched['answer']}")
            continue

        answer_text = resolve_worker_question(role, question, task, constraints)
        resolved.append({"question": question, "answer": answer_text})

    snippets = retrieve(f"{task} {' '.join(constraints)}")
    draft = write(task, snippets, constraints)

    result = review(task, draft)
    if not result.passed:
        print(f"[Organizer] 验收未通过，要求重写：{result.missing}")
        draft = write(task, snippets, constraints, feedback="; ".join(result.missing))
        result = review(task, draft)

    print("\n" + "=" * 60)
    print(draft)
    print("=" * 60)
    print(f"最终验收：{'通过' if result.passed else '未通过（带缺口交付）'}")
    return {"draft": draft, "passed": result.passed, "constraints": constraints}


def main() -> int:
    print("=" * 60)
    print("多 Agent 海龟汤协作 demo")
    print("=" * 60)
    print("示例任务：对比 LangGraph 和 CrewAI，给出一段简短结论")
    print("回答方式：是 / 否 / 补充说明；/exit 退出")
    print("=" * 60)

    try:
        task = input("\n任务 > ").strip()
    except (EOFError, KeyboardInterrupt):
        return 1
    if not task:
        return 2

    try:
        run(task)
    except UserExit:
        print("\n已退出")
        return 0
    except Exception as exc:
        print(f"\n[错误] {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
