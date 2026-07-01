import re
from typing import List, Dict, Tuple, Optional
from langchain.schema import Document


CITATION_PATTERN = re.compile(
    r"[《]"  # opening 《
    r"([^》]+)"  # law name
    r"[》]"  # closing 》
    r"\s*"
    r"(?:第"  # 第
    r"([一二三四五六七八九十百千\d]+)"  # number
    r"(?:[条章节条]|条之\d+)?"  # 条/章/节
    r")?"
)

LAW_ALIASES = {
    "海商法": "海商法",
    "海上交通安全法": "海上交通安全法",
    "海洋环境保护法": "海洋环境保护法",
    "船舶登记条例": "船舶登记条例",
    "船员条例": "船员条例",
    "船舶吨税法": "船舶吨税法",
    "国际海运条例": "国际海运条例",
    "港口法": "港口法",
    "内河交通安全管理条例": "内河交通安全管理条例",
    "防治船舶污染海洋环境管理条例": "防治船舶污染海洋环境管理条例",
    "SOLAS": "IMO Convention - SOLAS 详细",
    "MARPOL": "IMO Convention - MARPOL 详细",
    "STCW": "IMO Convention - STCW 详细",
    "ISM Code": "ISM Code",
    "MLC": "MLC 2006",
    "船舶安全管理规则": "ISM Code (国际安全管理规则)",
    "海事劳工公约": "MLC 2006 (海事劳工公约)",
}


def extract_citations(text: str) -> List[Dict[str, str]]:
    citations = []
    for m in CITATION_PATTERN.finditer(text):
        citations.append({
            "law": m.group(1),
            "article": m.group(2) if m.lastindex and m.group(2) else "",
            "full": m.group(0),
        })
    return citations


def _match_law(cited_law: str, source_name: str) -> bool:
    cited = cited_law.strip()
    if cited in source_name or source_name in cited:
        return True
    aliased = LAW_ALIASES.get(cited)
    if aliased and (aliased in source_name or source_name in aliased):
        return True
    return False


def _match_article(article: str, content: str) -> bool:
    if not article:
        return True
    patterns = [
        re.compile(re.escape(article) + r"\s*[条章节]"),
        re.compile(r"第?\s*" + re.escape(article) + r"\s*[条章节]"),
        re.compile(article),
    ]
    for p in patterns:
        if p.search(content):
            return True
    return False


def verify_citations(citations: List[Dict[str, str]], context: List[Document]) -> Dict[str, str]:
    results = {}
    for cit in citations:
        law = cit["law"]
        article = cit["article"]
        found = False
        for doc in context:
            content = doc.page_content
            src = doc.metadata.get("source", "")
            if _match_law(law, src) or _match_law(law, content):
                if _match_article(article, content):
                    found = True
                    break
        results[cit["full"]] = "verified" if found else "unverified"
    return results


def has_unverified(citations: List[Dict[str, str]], context: List[Document]) -> bool:
    result = verify_citations(citations, context)
    return any(v == "unverified" for v in result.values())


def get_unverified(citations: List[Dict[str, str]], context: List[Document]) -> List[str]:
    result = verify_citations(citations, context)
    return [k for k, v in result.items() if v == "unverified"]
