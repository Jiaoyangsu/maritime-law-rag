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


def classify_query(query: str) -> QueryPlan:
    q = query.strip()

    for pat, intent in FOLLOW_UP_PATTERNS:
        if pat.search(q):
            return QueryPlan(
                intent="follow_up",
                tools=[{"name": "retrieve", "params": {"query": q}}],
                original_query=query,
            )

    for pat, intent in CALC_PATTERNS:
        m = pat.search(q)
        if m:
            tonnage_match = re.search(r"(\d+)\s*(?:总吨|吨位|吨)", q)
            tonnage = float(tonnage_match.group(1)) if tonnage_match else 10000
            loss_type = "personal" if any(w in q for w in ["人身", "死亡", "伤亡", "personal"]) else "property"
            return QueryPlan(
                intent="calculation",
                tools=[{"name": "calculate", "params": {"tonnage": tonnage, "type_of_loss": loss_type}}],
                original_query=query,
            )

    summarize_pat = re.compile(r"^(概述|归纳|总结|介绍)\s*(.+)$")
    m = summarize_pat.match(q)
    if m:
        law = m.group(2).strip()
        if _is_law_name(law):
            return QueryPlan(
                intent="summarize",
                tools=[{"name": "summarize", "params": {"law": law}}],
                original_query=query,
            )

    what_is_pat = re.compile(r"^(.+?)(?:是什么|有哪些主要内容)$")
    m = what_is_pat.match(q)
    if m:
        law = m.group(1).strip()
        if _is_law_name(law):
            return QueryPlan(
                intent="summarize",
                tools=[{"name": "summarize", "params": {"law": law}}],
                original_query=query,
            )

    article_match = re.search(r"(第[一二三四五六七八九十百千\d]+[条章节])", q)
    if article_match:
        law_match = re.search(r"《([^》]+)》", q)
        if not law_match:
            law_match = re.search(r"^([^第]+?)第", q)
        if law_match:
            law = law_match.group(1).strip()
        else:
            law = ""
        return QueryPlan(
            intent="article_lookup",
            tools=[{"name": "get_article", "params": {"law": law, "article": article_match.group(1)}}],
            original_query=query,
        )

    if any(re.search(p, q) for p in COMPLEX_MARKERS):
        clean_q = re.sub(r"^(比较|对比|区别|分析|评估)", "", q).strip()
        parts = re.split(r"[和与、]", clean_q)
        laws = []
        for p in parts:
            name = _extract_law_name(p.strip())
            if name and _is_law_name(name):
                laws.append(name)
        laws = list(dict.fromkeys(laws))[:5]
        if len(laws) >= 2:
            topic = q
            for law in laws:
                topic = topic.replace(law, "").strip()
            topic = re.sub(r"^(比较|对比|区别|异同|分析|评估)", "", topic).strip().lstrip("的和与、")
            return QueryPlan(
                intent="comparison",
                tools=[{"name": "compare", "params": {"laws": ",".join(laws), "topic": topic or "相关规定"}}],
                original_query=query,
            )

    return QueryPlan(
        intent="general_retrieve",
        tools=[{"name": "retrieve", "params": {"query": q}}],
        original_query=query,
    )


def decompose_complex(query: str) -> List[str]:
    clauses = [c.strip() for c in re.split(r"[；;]", query) if c.strip()]
    return clauses or [query]
