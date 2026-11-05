# agents_learn_demo/rag_clarify.py
# ==============================================================================
# 最小 RAG + 海龟汤澄清 demo。
#
# 验证两件事：
#   1. 检索前：用是/否问题确认范围（年份、地区），再带着过滤条件检索；
#   2. 检索后：判断召回片段是否足够，不够再问一轮、重检索，仍不够则带缺口回答。
#
# 语料是本地 mock，检索用字符二元组相似度（不依赖 embedding 服务），
# 目的是演示澄清流程，而不是检索质量。
#
# 运行：
#   .\.venv\Scripts\python.exe agents_learn_demo\rag_clarify.py
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

MAX_PRE_QUESTIONS = 2
PRE_CONFIDENCE = 0.9
TOP_K = 3
RETRY_ATTEMPTS = 3

# ============ 本地语料 ============
DOCS = [
    {
        "id": "d1",
        "title": "差旅报销标准（2025 全国）",
        "year": "2025",
        "region": "全国",
        "content": "2025 年全国差旅报销标准：一线城市住宿每晚上限 600 元，二线城市 450 元；"
                   "高铁二等座实报实销；市内交通每天上限 80 元；餐补每天 100 元。",
    },
    {
        "id": "d2",
        "title": "差旅报销标准（2024 华东区）",
        "year": "2024",
        "region": "华东",
        "content": "2024 年华东区差旅标准：上海、杭州、南京住宿每晚上限 500 元；"
                   "高铁二等座实报实销；餐补每天 80 元。该版本自 2025 年起已被全国标准替代。",
    },
    {
        "id": "d3",
        "title": "差旅报销流程（2025）",
        "year": "2025",
        "region": "全国",
        "content": "差旅报销流程：出差结束后 10 个工作日内提交发票与行程单；"
                   "先由直属主管审批，再由财务复核；单笔超过 5000 元需部门负责人加签。",
    },
    {
        "id": "d4",
        "title": "远程办公政策（2025）",
        "year": "2025",
        "region": "全国",
        "content": "远程办公政策：每周最多 2 天远程；需提前一天在系统申请；"
                   "远程期间保持在线，核心会议须参加。",
    },
    {
        "id": "d5",
        "title": "采购审批流程（2025）",
        "year": "2025",
        "region": "全国",
        "content": "采购审批：单笔 1 万元以下由部门负责人审批；1 万至 5 万元由分管副总审批；"
                   "5 万元以上需总经理审批并附三家比价。",
    },
    {
        "id": "d6",
        "title": "信息安全规范（2024）",
        "year": "2024",
        "region": "全国",
        "content": "信息安全规范：禁止将公司数据上传到个人网盘；密码至少 12 位并每 90 天更换；"
                   "外部 U 盘使用前需登记。",
    },
    {
        "id": "d7",
        "title": "请假与调休规定（2025）",
        "year": "2025",
        "region": "全国",
        "content": "请假规定：年假需提前 3 个工作日申请；病假需提供医院证明；"
                   "加班可申请调休，调休需在 6 个月内使用。",
    },
]

# ============ 结构化输出模型 ============
class Filters(BaseModel):
    year: Optional[str] = Field(default=None, description="年份，如 2025；不确定时留空")
    region: Optional[str] = Field(default=None, description="地区，如 华东；不确定时留空")


class QueryClarification(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0, description="对问题范围的把握度")
    understanding: str = Field(min_length=1, max_length=300)
    filters: Filters = Field(default_factory=Filters)
    question: Optional[str] = Field(default=None, description="下一个是/否问题；范围清楚时留空")


class RetrievalCheck(BaseModel):
    sufficient: bool = Field(description="召回片段是否足以回答问题")
    confidence: float = Field(ge=0.0, le=1.0)
    missing: list[str] = Field(default_factory=list, description="还缺什么信息")
    question: Optional[str] = Field(default=None, description="不足时问用户的一个是/否问题（关于范围/版本）")


class RAGAnswer(BaseModel):
    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list, description="引用的文档 id")


# ============ 用户输入 ============
ASK: Callable[[str], str] = input
EXIT_WORDS = {"/exit", "exit", "quit", "q"}
DONE_WORDS = {"/done", "done", "直接回答"}


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
        return "用户要求直接回答"
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
            print(f"  [重试] {schema.__name__} 第 {attempt}/{RETRY_ATTEMPTS} 次：{type(exc).__name__}: {exc}")
    raise RuntimeError(f"{schema.__name__} 连续 {RETRY_ATTEMPTS} 次失败") from last_error


# ============ 检索 ============
def _normalize(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _bigrams(text: str) -> set[str]:
    return {text[i:i + 2] for i in range(len(text) - 1)}


def similarity(query: str, doc: dict) -> float:
    """查询覆盖率：50% 字符 + 50% 二元组出现在文档里的比例。"""
    a = _normalize(query)
    b = _normalize(doc["title"] + doc["content"])
    if not a:
        return 0.0
    chars = set(a)
    uni = len(chars & set(b)) / len(chars)
    grams = _bigrams(a)
    bi = len(grams & _bigrams(b)) / len(grams) if grams else 0.0
    return 0.5 * uni + 0.5 * bi


def retrieve(query: str, filters: Filters, top_k: int = TOP_K) -> tuple[list[tuple[float, dict]], bool]:
    """返回 (命中列表, 是否因为过滤条件太严而放宽)。"""
    def matches(doc: dict) -> bool:
        if filters.year and doc["year"] != filters.year:
            return False
        if filters.region and filters.region not in doc["region"]:
            return False
        return True

    filtered = [doc for doc in DOCS if matches(doc)]
    widened = False
    if not filtered:
        filtered = DOCS
        widened = True

    scored = [(similarity(query, doc), doc) for doc in filtered]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k], widened


# ============ 提示词 ============
PRE_SYSTEM = f"""你是检索前的澄清员。用是/否问题确认问题范围，重点是：
- 年份/版本（例如“2025 还是 2024”）
- 地区/适用范围（例如“全国还是华东”）
每轮只问一个封闭问题，用户可以补充文字。范围已经清楚时 confidence>=0.95 且 question 留空。
不要问文档内容本身能回答的事实，只问用户才知道的范围。"""

CHECK_SYSTEM = """你是检索质检员。判断给出的文档片段是否足以回答用户问题。
要求：能覆盖问题的关键点、版本一致、没有互相矛盾。
不足时给出一个关于范围/版本的是/否问题，让用户缩小范围；不要问文档里已有的内容。"""

ANSWER_SYSTEM = """你是回答生成器。只根据给定片段回答，引用用 [id] 标注。
如果片段不足以回答，明确说明缺什么，不要编造。"""


# ============ 主流程 ============
def run(question: str) -> Optional[dict]:
    print("\n[检索前澄清]")
    history: list = [HumanMessage(content=f"用户问题：{question}")]
    filters = Filters()
    pre_answers: list[str] = []

    for _ in range(MAX_PRE_QUESTIONS):
        result: QueryClarification = call_structured(
            QueryClarification,
            [SystemMessage(content=PRE_SYSTEM), *history],
        )
        filters = result.filters or filters
        print(f"  理解：{result.understanding}（{result.confidence:.0%}）")
        if result.filters.year or result.filters.region:
            print(f"  过滤条件：year={result.filters.year or '-'} region={result.filters.region or '-'}")
        if result.confidence >= PRE_CONFIDENCE or not result.question:
            break
        print(f"  问：{result.question}")
        answer = ask_user("你（是/否/补充）> ")
        pre_answers.append(answer)
        history.append(AIMessage(content=result.question))
        history.append(HumanMessage(content=answer))

    search_text = question + " " + " ".join(pre_answers)
    hits, widened = retrieve(search_text, filters)
    print(f"\n[检索] 命中 {len(hits)} 条" + ("（过滤条件太严，已放宽）" if widened else ""))
    for score, doc in hits:
        print(f"  [{doc['id']}] {doc['title']}（相似度 {score:.2f}）")

    context = "\n".join(
        f"[{doc['id']}] {doc['title']}（{doc['year']}/{doc['region']}）\n{doc['content']}"
        for _, doc in hits
    )

    # 检索后判定
    check: RetrievalCheck = call_structured(
        RetrievalCheck,
        [
            SystemMessage(content=CHECK_SYSTEM),
            HumanMessage(
                content=f"""问题：{question}
范围：year={filters.year or '未指定'} region={filters.region or '未指定'}

召回片段：
{context}

是否足够回答？不够就给出一个范围问题是/否问题。"""
            ),
        ],
    )
    print(f"\n[检索后判定] sufficient={check.sufficient}, confidence={check.confidence:.0%}")

    if not check.sufficient and check.question:
        print(f"  问：{check.question}")
        answer = ask_user("你（是/否/补充）> ")
        search_text = f"{search_text} {check.question} {answer}"
        hits, _ = retrieve(search_text, filters)
        print(f"\n[重检索] 命中 {len(hits)} 条")
        for score, doc in hits:
            print(f"  [{doc['id']}] {doc['title']}（相似度 {score:.2f}）")
        context = "\n".join(
            f"[{doc['id']}] {doc['title']}（{doc['year']}/{doc['region']}）\n{doc['content']}"
            for _, doc in hits
        )
        check = call_structured(
            RetrievalCheck,
            [
                SystemMessage(content=CHECK_SYSTEM),
                HumanMessage(
                    content=f"""问题：{question}
补充信息：{answer}
召回片段：
{context}

是否足够回答？只判断，不要反问我。"""
                ),
            ],
        )
        print(f"[检索后判定] sufficient={check.sufficient}, confidence={check.confidence:.0%}")

    gap_note = ""
    if not check.sufficient:
        gap_note = f"\n注意：检索结果可能不充分，缺失：{'; '.join(check.missing) or '未说明'}。"

    result: RAGAnswer = call_structured(
        RAGAnswer,
        [
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(
                content=f"""问题：{question}
范围：year={filters.year or '未指定'} region={filters.region or '未指定'}

文档片段：
{context}
{gap_note}

请回答，并用 [id] 标注引用。"""
            ),
        ],
    )

    print("\n" + "=" * 60)
    print(result.answer)
    if result.citations:
        print(f"\n引用：{', '.join(result.citations)}")
    print("=" * 60)
    return {"answer": result.answer, "citations": result.citations, "filters": filters}


def main() -> int:
    print("=" * 60)
    print("最小 RAG + 海龟汤澄清")
    print("=" * 60)
    print("示例问题：出差住宿费能报多少？")
    print("回答方式：是 / 否 / 补充说明；/exit 退出")
    print("=" * 60)

    try:
        question = input("\n问题 > ").strip()
    except (EOFError, KeyboardInterrupt):
        return 1
    if not question:
        return 2

    try:
        run(question)
    except UserExit:
        print("\n已退出")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
