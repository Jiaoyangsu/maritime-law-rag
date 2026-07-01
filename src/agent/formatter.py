from typing import List, Tuple, Dict, Any
from langchain.schema import Document
from src.config import TOP_K


def format_results(results: List[Tuple[Document, float]], max_preview: int = 5) -> str:
    if not results:
        return "未检索到相关法条。"
    lines = [f"检索到 {len(results)} 条相关法条：\n"]
    for i, (doc, score) in enumerate(results[:max_preview], 1):
        src = doc.metadata.get("source", "未知")
        pid = doc.metadata.get("parent_id", "?")
        content = doc.page_content[:200].strip().replace("\n", " ")
        lines.append(f"[{i}] 《{src}》 (相关度: {score:.4f})")
        lines.append(f"    {content}...")
    return "\n".join(lines)


def format_comparison(result: Dict[str, List[Tuple[Document, float]]]) -> str:
    if not result:
        return "无法进行比较，请确认法律名称。"
    lines = ["【法律比较结果】\n"]
    for law, docs in result.items():
        lines.append(f"▸ 《{law}》 - {len(docs)} 条相关段落")
        for i, (doc, score) in enumerate(docs[:3], 1):
            text = doc.page_content[:100].strip().replace("\n", " ")
            lines.append(f"  {i}. {text}...")
        lines.append("")
    return "\n".join(lines)


def format_calculation(calc_result: Dict[str, Any]) -> str:
    if not calc_result:
        return "无法计算，请提供正确的船舶吨位。"
    lines = ["【海事赔偿责任限额计算结果】\n"]
    lines.append(f"  船舶总吨位: {calc_result['tonnage']:,.0f} 总吨")
    lines.append(f"  损失类型: {calc_result['loss_type']}")
    lines.append(f"  责任限额 (SDR): {calc_result['limit_sdr']:,.0f}")
    lines.append(f"  约合人民币: ¥{calc_result['limit_approx_cny']:,.0f}")
    lines.append(f"\n  备注: {calc_result['note']}")
    return "\n".join(lines)


def format_article(article_results: List[Tuple[Document, float]]) -> str:
    if not article_results:
        return "未找到匹配的条款。"
    lines = ["【精确条款查询结果】\n"]
    for doc, score in article_results:
        src = doc.metadata.get("source", "未知")
        lines.append(f"《{src}》")
        lines.append(doc.page_content.strip())
        lines.append("")
    return "\n".join(lines)


def format_summary(summary: str) -> str:
    if not summary:
        return "无法生成概述。"
    return f"【法律概述】\n\n{summary}"


def format_agent_step(step: str, detail: str = "") -> str:
    if detail:
        return f"[{step}] {detail}"
    return f"[{step}]"


def format_disclaimer() -> str:
    return "\n[免责声明] 以上信息仅供参考，不构成法律意见。具体法律适用请咨询专业律师。"
