# agents_learn_demo/peer_agent_clarify.py
# ==============================================================================
# 平级多 Agent 海龟汤协作 demo（没有组织者）。
#
# 三个职责不同的 agent 平级协作：
#   Retrieval：查资料
#   Writing：写结论
#   Reviewer：验收
#
# 每个 agent 在自己动手前，可以向另外两个 agent 提一个“是/否”问题；
# 对方回答 yes/no/unknown + 证据；答不上来时由第三个 agent 仲裁，仍不确定再问用户。
#
# 关键约束（防止平级协作退化成互相客套）：
#   1. 全局提问预算 MAX_PEER_QUESTIONS，用尽后停止澄清、带缺口继续；
#   2. 重复问题查黑板复用，不再重复问；
#   3. 回答必须带证据，没有证据只能 unknown；
#   4. 仲裁顺序：证据 → 第三方投票 → 用户。
#
# 运行：
#   .\.venv\Scripts\python.exe agents_learn_demo\peer_agent_clarify.py
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
MAX_PEER_QUESTIONS = 3
MAX_USER_ESCALATIONS = 1   # 平级协作里最多打断用户几次，避免变成连环追问
ROLES = {
    "retrieval": "你负责检索本地资料库，知道 d1/d2/d3/d4 四篇文档的内容，可以回答资料相关问题。",
    "writing": "你负责把资料写成不超过 200 字的结论，知道当前草稿和表达方式。",
    "reviewer": "你负责验收，知道验收标准：覆盖对象、有明确结论、引用资料编号。",
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
class PeerAction(BaseModel):
    target: Literal["retrieval", "writing", "reviewer", "none"] = Field(
        description="向谁提问；none 表示没有问题"
    )
    question: Optional[str] = Field(default=None, description="一个能用是/否回答的问题")
    reason: str = Field(default="", max_length=200)


class PeerAnswer(BaseModel):
    answer: Literal["yes", "no", "unknown"] = Field(description="能否确定；不能确定就 unknown")
    evidence: str = Field(default="", max_length=400, description="回答依据；unknown 时留空")


class Arbitration(BaseModel):
    answer: Literal["yes", "no", "unknown"] = Field(description="根据证据给出的判断")
    rationale: str = Field(default="", max_length=300)


class ReviewCheck(BaseModel):
    question: str
    passed: bool
    evidence: str = ""


class Review(BaseModel):
    checks: list[ReviewCheck] = Field(min_length=1, max_length=6)
    passed: bool
    missing: list[str] = Field(default_factory=list)


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


# ============ 文本工具 ============
def _normalize(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _bigrams(text: str) -> set[str]:
    return {text[i:i + 2] for i in range(len(text) - 1)}


def questions_similar(left: str, right: str, threshold: float = 0.5) -> bool:
    ga, gb = _bigrams(_normalize(left)), _bigrams(_normalize(right))
    if not ga or not gb:
        return False
    return len(ga & gb) / len(ga | gb) >= threshold


def format_blackboard(blackboard: list[dict]) -> str:
    if not blackboard:
        return "（暂无问答记录）"
    lines = []
    for item in blackboard:
        if item["type"] == "question":
            lines.append(
                f"- {item['asker']} → {item['answerer']}：{item['question']} "
                f"→ {item['answer']}（{item.get('evidence', '')}）"
            )
        else:
            lines.append(f"- [仲裁] {item['question']} → {item['answer']}")
    return "\n".join(lines)


# ============ 平级提问协议 ============
def peer_ask(
    asker: str,
    task: str,
    context: str,
    blackboard: list[dict],
) -> Optional[dict]:
    """asker 决定是否向某个平级 agent 提一个封闭问题。"""
    peers = [role for role in ROLES if role != asker]
    if len([item for item in blackboard if item["type"] == "question"]) >= MAX_PEER_QUESTIONS:
        print(f"[{asker}] 提问预算已用尽，停止澄清，按现有信息继续。")
        return None

    action: PeerAction = call_structured(
        PeerAction,
        [
            SystemMessage(
                content=(
                    f"{ROLES[asker]} 你是平级协作中的一员，没有上级。"
                    f"如果需要确认接口或信息，可以向 {peers} 中的一个提一个是/否问题；"
                    "没有问题就 target=none。不要问自己能查到的事，不要重复黑板上已有的问题。"
                )
            ),
            HumanMessage(
                content=f"""任务：{task}

当前进展：
{context}

黑板（已有问答）：
{format_blackboard(blackboard)}

请决定是否提问。"""
            ),
        ],
    )

    if action.target == "none" or not action.question:
        print(f"[{asker}] 没有需要确认的问题。")
        return None
    if action.target == asker:
        print(f"[{asker}] 目标无效，跳过。")
        return None

    question = action.question.strip()
    duplicate = next(
        (item for item in blackboard if item["type"] == "question" and questions_similar(question, item["question"])),
        None,
    )
    if duplicate:
        print(f"[{asker}] 问题与黑板重复，直接复用：{duplicate['answer']}")
        return duplicate

    print(f"[{asker} → {action.target}] {question}（原因：{action.reason}）")
    answer: PeerAnswer = call_structured(
        PeerAnswer,
        [
            SystemMessage(
                content=(
                    f"{ROLES[action.target]} 回答 {asker} 的问题。"
                    "只根据你知道的信息回答；不确定就 answer=unknown，不要猜。"
                    "回答 yes/no 时必须给出 evidence。"
                )
            ),
            HumanMessage(content=f"任务：{task}\n问题：{question}\n黑板：\n{format_blackboard(blackboard)}"),
        ],
    )

    record = {
        "type": "question",
        "asker": asker,
        "answerer": action.target,
        "question": question,
        "answer": answer.answer,
        "evidence": answer.evidence,
    }

    if answer.answer == "unknown":
        record = arbitrate(record, task, blackboard)

    print(f"[{action.target} → {asker}] {record['answer']}（{record.get('evidence', '')}）")
    blackboard.append(record)
    return record


def arbitrate(record: dict, task: str, blackboard: list[dict]) -> dict:
    """回答 unknown 时，让第三个 agent 根据证据投票，仍不确定再问用户。"""
    asker, answerer = record["asker"], record["answerer"]
    third = next(role for role in ROLES if role not in (asker, answerer))
    print(f"[仲裁] {answerer} 回答 unknown，由 {third} 根据证据判断。")

    verdict: Arbitration = call_structured(
        Arbitration,
        [
            SystemMessage(
                content=(
                    f"{ROLES[third]} 你是第三方仲裁者。根据黑板里的证据判断这个问题该答 yes 还是 no；"
                    "证据不足就 answer=unknown。"
                )
            ),
            HumanMessage(
                content=f"""任务：{task}
争议问题：{record['question']}
黑板：
{format_blackboard(blackboard)}

请给出仲裁结论。"""
            ),
        ],
    )
    if verdict.answer != "unknown":
        record["answer"] = verdict.answer
        record["evidence"] = verdict.rationale
        record["arbitrated_by"] = third
        blackboard.append(
            {"type": "arbitration", "question": record["question"], "answer": verdict.answer, "by": third}
        )
        return record

    used = sum(1 for item in blackboard if item.get("by") == "user")
    if used >= MAX_USER_ESCALATIONS:
        print("[仲裁] 用户升级次数已用尽，记录 unknown，按现有信息继续。")
        record["answer"] = "unknown"
        record["evidence"] = "用户升级预算用尽"
        blackboard.append(
            {"type": "arbitration", "question": record["question"], "answer": "unknown", "by": "budget"}
        )
        return record

    print("[仲裁] 仍无法确定，升级问用户。")
    user_answer = ask_user(f"关于「{record['question']}」，你（是/否/补充）> ")
    record["answer"] = user_answer
    record["evidence"] = "用户确认"
    record["arbitrated_by"] = "user"
    blackboard.append(
        {"type": "arbitration", "question": record["question"], "answer": user_answer, "by": "user"}
    )
    return record


# ============ 三个 agent 的工作 ============
def retrieval_work(task: str, blackboard: list[dict]) -> list[dict]:
    print("\n[Retrieval] 检索资料...")
    peer_ask("retrieval", task, "准备检索资料，尚未开始", blackboard)
    query = _normalize(task + " " + " ".join(item.get("answer", "") for item in blackboard))
    scored = []
    for doc in DOCS:
        doc_text = _normalize(doc["title"] + doc["content"])
        score = len(set(query) & set(doc_text)) / max(1, len(set(query)))
        scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    hits = [doc for score, doc in scored[:2]]
    for doc in hits:
        print(f"  命中 [{doc['id']}] {doc['title']}")
    return hits


def writing_work(task: str, docs: list[dict], blackboard: list[dict], feedback: str = "") -> str:
    print("\n[Writing] 撰写结论...")
    peer_ask("writing", task, f"已拿到资料：{[d['id'] for d in docs]}", blackboard)
    material = "\n".join(f"[{doc['id']}] {doc['content']}" for doc in docs) or "（无资料）"
    prompt = f"""任务：{task}

资料：
{material}

要求：不超过 200 字，必须引用资料编号，例如 [d1]。"""
    if feedback:
        prompt += f"\n上一版问题：{feedback}，请修正。"
    text = call_text([SystemMessage(content="你是写作 agent，只根据资料写，不要编造。"), HumanMessage(content=prompt)])
    print(f"  草稿：{text[:70]}...")
    return text


def reviewer_work(task: str, draft: str, blackboard: list[dict]) -> Review:
    print("\n[Reviewer] 验收...")
    peer_ask("reviewer", task, f"草稿：{draft[:120]}", blackboard)
    result: Review = call_structured(
        Review,
        [
            SystemMessage(
                content=(
                    "你是审查 agent。用 3 条二元检查验收：覆盖对象、有明确结论、引用资料编号。"
                    "每条给 passed 和 evidence，最后给整体 passed 与 missing。"
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
    print("平级多 Agent 海龟汤协作（无组织者）")
    print("=" * 60)
    print(f"任务：{task}\n")

    blackboard: list[dict] = []

    docs = retrieval_work(task, blackboard)
    draft = writing_work(task, docs, blackboard)
    result = reviewer_work(task, draft, blackboard)

    if not result.passed:
        print(f"[Writing] 根据验收意见重写：{result.missing}")
        draft = writing_work(task, docs, blackboard, feedback="; ".join(result.missing))
        result = reviewer_work(task, draft, blackboard)

    print("\n" + "=" * 60)
    print(draft)
    print("=" * 60)
    print(f"最终验收：{'通过' if result.passed else '未通过（带缺口交付）'}")
    print("\n协作黑板：")
    print(format_blackboard(blackboard))

    questions = [item for item in blackboard if item["type"] == "question"]
    print(f"\n提问次数：{len(questions)}；预算：{MAX_PEER_QUESTIONS}")
    return {"draft": draft, "passed": result.passed, "blackboard": blackboard}


def main() -> int:
    print("=" * 60)
    print("平级多 Agent 协作 demo")
    print("=" * 60)
    print("示例任务：对比 LangGraph 和 CrewAI，给出一段简短结论")
    print("/exit 退出")
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
