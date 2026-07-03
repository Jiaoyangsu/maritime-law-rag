import re
from typing import List, Dict, Optional

KNOWN_LAWS = [
    "海商法", "海上交通安全法", "海洋环境保护法",
    "船舶登记条例", "船员条例", "船舶吨税法",
    "国际海运条例", "港口法", "内河交通安全管理条例",
    "防治船舶污染海洋环境管理条例",
    "SOLAS", "MARPOL", "STCW", "ISM", "MLC",
]

COMPLEX_MARKERS = [
    "比较", "对比", "区别", "异同", "vs", "versus",
    "以及", "和.*的.*区别", "哪个", "有什么不同",
    "分析", "评估", "综合",
]

FOLLOW_UP_PATTERNS = [
    (re.compile(r"^第[一二三四五六七八九十百千\d]+条"), "article_lookup"),
    (re.compile(r"^它|^这|^那|^其|^上述|^以上"), "follow_up"),
    (re.compile(r"^(详细|具体|进一步|继续|还有|另外)"), "follow_up"),
]

CALC_PATTERNS = [
    (re.compile(r"(\d+)\s*(?:总吨|吨位|吨)"), "calc"),
    (re.compile(r"(责任限额|赔偿限额|赔偿责任限制)"), "calc"),
]

MULTI_PART_SPLITTERS = re.compile(r"[；;]|(?:并且|同时|以及|另外|此外)")


class QueryPlan:
    def __init__(self, intent: str, tools: List[Dict], original_query: str):
        self.intent = intent
        self.tools = tools
        self.original_query = original_query

    def __repr__(self):
        return f"QueryPlan(intent={self.intent}, tools={self.tools})"


def _is_law_name(word: str) -> bool:
    word = word.strip("《》").strip()
    return any(law in word or word in law for law in KNOWN_LAWS)


def _extract_law_name(text: str) -> str:
    for law in KNOWN_LAWS:
        if law in text:
            return law
    return text.strip("《》").strip()


def _has_calc_query(text: str) -> bool:
    return bool(re.search(r"(责任限额|赔偿限额|赔偿责任限制)", text))


def _extract_tonnage(text: str) -> float:
    m = re.search(r"(\d+)\s*(?:总吨|吨位|吨)", text)
    return float(m.group(1)) if m else 10000


def _detect_loss_type(text: str) -> str:
    return "personal" if any(w in text for w in ["人身", "死亡", "伤亡", "personal"]) else "property"


def _build_retrieve_tool(query: str) -> Dict:
    return {"name": "retrieve", "params": {"query": query}}


def _build_calc_tool(text: str) -> Dict:
    tonnage = _extract_tonnage(text)
    loss_type = _detect_loss_type(text)
    return {"name": "calculate", "params": {"tonnage": tonnage, "type_of_loss": loss_type}}


def _build_article_tool(text: str) -> Optional[Dict]:
    article_match = re.search(r"(第[一二三四五六七八九十百千\d]+[条章节])", text)
    if not article_match:
        return None
    law_match = re.search(r"《([^》]+)》", text)
    if not law_match:
        m = re.search(r"^([^第]+?)第", text)
        if m:
            law_match = m
    law = law_match.group(1).strip() if law_match else ""
    return {"name": "get_article", "params": {"law": law, "article": article_match.group(1)}}


def _build_compare_tool(text: str) -> Optional[Dict]:
    clean_q = re.sub(r"^(比较|对比|区别|分析|评估)", "", text).strip()
    parts = re.split(r"[和与、]", clean_q)
    laws = []
    for p in parts:
        name = _extract_law_name(p.strip())
        if name and _is_law_name(name):
            laws.append(name)
    laws = list(dict.fromkeys(laws))[:5]
    if len(laws) >= 2:
        topic = text
        for law in laws:
            topic = topic.replace(law, "").strip()
        topic = re.sub(r"^(比较|对比|区别|异同|分析|评估)", "", topic).strip().lstrip("的和与、")
        return {"name": "compare", "params": {"laws": ",".join(laws), "topic": topic or "相关规定"}}
    return None


def _classify_single(q: str) -> Optional[QueryPlan]:
    for pat, intent in FOLLOW_UP_PATTERNS:
        if pat.search(q):
            return QueryPlan(
                intent="follow_up",
                tools=[_build_retrieve_tool(q)],
                original_query=q,
            )

    for pat, _ in CALC_PATTERNS:
        m = pat.search(q)
        if m:
            return QueryPlan(
                intent="calculation",
                tools=[_build_calc_tool(q)],
                original_query=q,
            )

    summarize_pat = re.compile(r"^(概述|归纳|总结|介绍)\s*(.+)$")
    m = summarize_pat.match(q)
    if m:
        law = m.group(2).strip()
        if _is_law_name(law):
            return QueryPlan(
                intent="summarize",
                tools=[{"name": "summarize", "params": {"law": law}}],
                original_query=q,
            )

    what_is_pat = re.compile(r"^(.+?)(?:是什么|有哪些主要内容)$")
    m = what_is_pat.match(q)
    if m:
        law = m.group(1).strip()
        if _is_law_name(law):
            return QueryPlan(
                intent="summarize",
                tools=[{"name": "summarize", "params": {"law": law}}],
                original_query=q,
            )

    article_tool = _build_article_tool(q)
    if article_tool:
        return QueryPlan(
            intent="article_lookup",
            tools=[article_tool],
            original_query=q,
        )

    compare_tool = _build_compare_tool(q)
    if compare_tool:
        return QueryPlan(
            intent="comparison",
            tools=[compare_tool],
            original_query=q,
        )

    return None


def classify_query(query: str) -> QueryPlan:
    q = query.strip()

    qa_pat = re.compile(r"^(.+?)[的之](.+?)是(什么|如何|多少|哪些)")
    multi_article_pat = re.compile(r"(第[^第]+?和第[^第]+?[条章节])")

    parts = [p.strip() for p in MULTI_PART_SPLITTERS.split(q) if p and p.strip()]
    if len(parts) < 2:
        parts = [q]

    tools = []
    combined_intent = "general_retrieve"

    for part in parts:
        single = _classify_single(part)
        if single:
            for t in single.tools:
                tools.append(t)
            if combined_intent == "general_retrieve":
                combined_intent = single.intent
        else:
            tools.append(_build_retrieve_tool(part))

    if len(tools) >= 2 and combined_intent != "comparison":
        combined_intent = "multi_step"

    if tools:
        return QueryPlan(
            intent=combined_intent,
            tools=tools,
            original_query=q,
        )

    return QueryPlan(
        intent="general_retrieve",
        tools=[_build_retrieve_tool(q)],
        original_query=q,
    )


def decompose_complex(query: str) -> List[str]:
    parts = [c.strip() for c in MULTI_PART_SPLITTERS.split(query) if c.strip()]
    return parts or [query]
