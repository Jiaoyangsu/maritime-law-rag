import re
from typing import List, Tuple, Optional, Dict, Any
from langchain.schema import Document
from src.vector_store.store import get_store
from src.config import TOP_K

TOOL_DESCRIPTIONS = {
    "retrieve": {
        "description": "语义检索法律条文。输入自然语言查询，返回相关法条段落。可用于任何法律问题。",
        "params": {"query": "检索问句 (必填)", "k": "返回结果数量 (默认5)"},
    },
    "get_article": {
        "description": "精确查找某部法律的特定条款。适用于需要精确法条编号的场景。",
        "params": {"law": "法律名称（如'海商法'、'SOLAS'）", "article": "条款编号（如'第一百六十九条'、'第12条'）"},
    },
    "compare": {
        "description": "比较多部法律在特定主题上的规定。",
        "params": {"laws": "法律名称列表（逗号分隔）", "topic": "比较主题"},
    },
    "calculate": {
        "description": "根据船舶吨位计算海事赔偿责任限额。",
        "params": {"tonnage": "船舶总吨位（数值）", "type_of_loss": "损失类型: personal/property"},
    },
    "summarize": {
        "description": "归纳某部法律的核心内容和主要条款。",
        "params": {"law": "法律名称"},
    },
}


def retrieve(query: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
    store = get_store()
    results = store.hybrid_search(query, k=k)
    return results


def get_article(law: str, article: str) -> List[Tuple[Document, float]]:
    store = get_store()

    # 策略1: 按 source 元数据筛出该法律所有 parent chunk，精确匹配条款编号
    parents = store.get_parents_by_source(law)
    pattern = re.compile(re.escape(article))
    exact_matches = []
    for pid, doc in parents.items():
        if pattern.search(doc.page_content):
            exact_matches.append((doc, 1.0))
    if exact_matches:
        return exact_matches[:5]

    # 策略2: BM25 搜索条款编号（jieba 分词对中文编号匹配更好），按 law 过滤
    bm25_results = store.bm25_search(f"{law} {article}", k=50)
    bm25_matches = []
    for doc, score in bm25_results:
        src = doc.metadata.get("source", "")
        if law.lower() not in src.lower():
            continue
        if pattern.search(doc.page_content):
            bm25_matches.append((doc, float(score)))
    if bm25_matches:
        return bm25_matches[:5]

    # 策略3: hybrid search 兜底（原逻辑）
    candidates = store.hybrid_search(f"{law} {article}", k=50)
    fallback = []
    for doc, score in candidates:
        src = doc.metadata.get("source", "")
        if law.lower() not in src.lower():
            continue
        if pattern.search(doc.page_content):
            fallback.append((doc, score))
    return fallback[:5]


def compare(laws: str, topic: str) -> Dict[str, List[Tuple[Document, float]]]:
    law_list = [l.strip() for l in laws.split(",")]
    result = {}
    for law in law_list:
        query = f"{law} {topic}"
        docs = retrieve(query, k=3)
        result[law] = docs
    return result


def calculate(tonnage: float, type_of_loss: str = "property") -> Optional[Dict[str, Any]]:
    if tonnage <= 0:
        return None
    type_of_loss = type_of_loss.lower()
    units_per_ton = 0
    if type_of_loss in ("personal", "人身"):
        if tonnage <= 2000:
            limit = 2000 * 167000
        else:
            limit = 2000 * 167000 + (tonnage - 2000) * 16700
        return {
            "tonnage": tonnage,
            "loss_type": "人身伤亡",
            "limit_sdr": limit,
            "limit_approx_cny": round(limit * 9.5),
            "note": "估算值，按《海商法》第二百一十条。实际限额可能因汇率、船型调整。",
        }
    else:
        if tonnage <= 2000:
            limit = 2000 * 167000
        elif tonnage <= 30000:
            limit = 2000 * 167000 + (tonnage - 2000) * 16700
        elif tonnage <= 70000:
            limit = 2000 * 167000 + 28000 * 16700 + (tonnage - 30000) * 12600
        else:
            limit = 2000 * 167000 + 28000 * 16700 + 40000 * 12600 + (tonnage - 70000) * 4100
        return {
            "tonnage": tonnage,
            "loss_type": "财产损失",
            "limit_sdr": limit,
            "limit_approx_cny": round(limit * 9.5),
            "note": "估算值，按《海商法》第二百一十条。实际限额可能因汇率、船型调整。",
        }


def summarize(law: str) -> Optional[str]:
    store = get_store()
    parents = store.get_parents_by_source(law)
    if not parents:
        return None
    texts = [doc.page_content for doc in parents.values()]
    total = sum(len(t) for t in texts)
    return f"[{law}] 共 {len(parents)} 个段落, 总计 {total} 字符。包含以下核心内容:\n" + "\n".join(
        f"  - {t[:80].strip()}..." for t in texts[:10]
    )


RETRY_SYNONYMS_FALLBACK = [
    "碰撞:触碰", "救助:救援", "赔偿:补偿", "责任:义务",
    "船舶:船", "船员:海员", "港口:港", "运输:运送",
    "安全:保安", "污染:排放", "保险:保赔", "证书:许可",
    "所有人:船东", "承运人:运输人", "托运人:发货人",
    " maritime:ship", "ship:vessel",
]


TOOL_FUNCTIONS = {
    "retrieve": retrieve,
    "get_article": get_article,
    "compare": compare,
    "calculate": calculate,
    "summarize": summarize,
}
