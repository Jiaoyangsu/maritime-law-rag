import re
from typing import Optional

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

LAW_NAME_PATTERN = re.compile(
    r"(?:中华(?:人民共和(?:国|园))?)?"
    r"(海商法"
    r"|海上交通安全法"
    r"|海洋环境保护法"
    r"|船舶登记条例"
    r"|船员条例"
    r"|船舶吨税法"
    r"|国际海运条例"
    r"|港口法"
    r"|内河交通安全管理条例"
    r"|防治船舶污染海洋环境管理条例)"
)

ABBREVIATION_MAP = {
    "海商": "海商法",
    "海上交通": "海上交通安全法",
    "海洋环境": "海洋环境保护法",
    "船舶登记": "船舶登记条例",
    "船员": "船员条例",
    "船舶吨税": "船舶吨税法",
    "国际海运": "国际海运条例",
    "港口": "港口法",
    "内河交通": "内河交通安全管理条例",
    "防治船舶污染": "防治船舶污染海洋环境管理条例",
}

TECHNICAL_SYNONYMS = {
    "赔偿": ["赔偿", "补偿", "赔付"],
    "碰撞": ["碰撞", "触碰", "撞击"],
    "救助": ["救助", "救援", "施救"],
    "责任": ["责任", "义务"],
    "船舶": ["船舶", "船"],
    "船员": ["船员", "海员"],
    "港口": ["港口", "港"],
    "污染": ["污染", "排放"],
    "保险": ["保险", "保赔"],
    "证书": ["证书", "许可"],
    "所有人": ["所有人", "船东"],
    "承运人": ["承运人", "运输人"],
    "托运人": ["托运人", "发货人"],
    "时效": ["时效", "诉讼时效", "期间"],
    "限额": ["限额", "限制", "赔偿限额", "责任限额"],
    "安全": ["安全", "保安"],
    "运输": ["运输", "运送"],
}

FOLLOW_UP_CLEANUP = re.compile(r"^(它|这|那|其|上述|以上|该)\s*")
FILLER_CLEANUP = re.compile(r"(请问|我想问|我想知道|请告诉我|能不能|可否|是否)")
ARTICLE_CLEANUP = re.compile(r"(?:第\s*)([一二三四五六七八九十百千\d]+)\s*(条|章|节)")


def rewrite_llm(query: str, context: str = "") -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import SystemMessage, HumanMessage

        kwargs = {"model": OPENAI_MODEL, "api_key": OPENAI_API_KEY, "temperature": 0}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        llm = ChatOpenAI(**kwargs)

        prompt = f"""你是一名海事法律检索助手。你的任务是将用户的自然语言问题改写成更适合法律条文检索的形式。

## 改写规则
1. 补全法律名称简称（"海商法" → "中华人民共和国海商法"）
2. 在保持核心法律术语不变的前提下，移除口语化表达
3. 如果用户之前提到过某部法律，保留该法律名称
4. 输出**仅**包含改写后的查询语句，不要任何解释

## 示例
用户: 碰撞了怎么赔
改写: 船舶碰撞 赔偿 责任 海商法

用户: 第175条说啥
改写: 海商法 第一百七十五条 船舶碰撞 船长义务

用户: 船东能限制多少赔偿
改写: 船舶所有人 责任限制 赔偿限额 海商法 海事赔偿责任限制

用户: 船员有啥权利
改写: 船员 权利 船员条例

## 当前查询
用户: {query}"""

        if context:
            prompt += f"\n\n历史对话上下文:\n{context}\n请结合上下文改写。"

        messages = [
            SystemMessage(content="你是一个法律检索查询改写助手。只输出改写后的查询。"),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        rewritten = response.content.strip().strip('"\'')
        if rewritten and len(rewritten) > 3:
            return rewritten
    except Exception:
        pass
    return None


def rewrite_rule(query: str) -> str:
    q = query.strip()

    q = FILLER_CLEANUP.sub("", q)
    q = FOLLOW_UP_CLEANUP.sub("", q)

    for abbr, full in ABBREVIATION_MAP.items():
        if abbr in q and full not in q:
            q = f"{full} {q}"

    article_match = ARTICLE_CLEANUP.search(q)
    if article_match and "海商法" not in q and "法" not in q:
        q = f"海商法 {q}"

    expanded_terms = []
    for term, synonyms in TECHNICAL_SYNONYMS.items():
        if term in q:
            expanded_terms.extend(synonyms)

    if expanded_terms:
        unique = list(dict.fromkeys(expanded_terms))
        q = f"{q} {' '.join(unique)}"

    parts = [p.strip() for p in q.split() if p.strip()]
    return " ".join(dict.fromkeys(parts))


def rewrite_query(query: str, context: str = "") -> str:
    llm_result = rewrite_llm(query, context)
    if llm_result:
        return llm_result
    return rewrite_rule(query)
