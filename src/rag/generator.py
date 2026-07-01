from typing import List, Optional
from src.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

SYSTEM_PROMPT = """你是一名精通中国及国际海事法律法规的资深法律顾问。你的任务是基于提供的法律条文，准确、专业地回答用户的问题。

## 核心规则
1. **只引用检索到的法条**：仅使用下方"参考法条"部分提供的内容作答，不得编造或添加未提供的法律条款。
2. **明确引用法条来源**：每次引用时必须标注具体法律名称和条款编号，例如"根据《中华人民共和国海商法》第一百六十九条"。
3. **无法回答时明确告知**：如果检索到的法条不足以回答用户问题，请明确告知"根据现有检索结果，无法完整回答该问题"，并提供最相关的部分信息。
4. **多法条综合分析**：当问题涉及多个法律领域时，综合引用相关法条，说明它们之间的关系。
5. **保持专业、准确、简洁**：使用规范的法学用语，回答结构清晰，避免冗余。

## 回答格式
- 先直接回答问题
- 然后列出支持的法条依据
- 最后可附加必要的解释或注意事项"""


def _build_llm():
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            return None
        try:
            from langchain_openai import ChatOpenAI
            kwargs = {"model": OPENAI_MODEL, "api_key": OPENAI_API_KEY, "temperature": 0}
            if OPENAI_BASE_URL:
                kwargs["base_url"] = OPENAI_BASE_URL
            return ChatOpenAI(**kwargs)
        except ImportError:
            return None
    elif LLM_PROVIDER == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
        except ImportError:
            return None
    return None


def build_prompt(query: str, context_chunks: List[str], sources: Optional[List[str]] = None) -> str:
    context = "\n\n---\n\n".join(
        f"[来源: {src or '未知'}]\n{chunk}" if src else chunk
        for chunk, src in zip(context_chunks, sources or [None] * len(context_chunks))
    )
    return f"""参考法条：
{context}

---
用户问题：{query}

请基于以上参考法条回答用户的问题。"""


def generate_answer(
    query: str,
    context_chunks: List[str],
    sources: Optional[List[str]] = None,
) -> Optional[str]:
    llm = _build_llm()
    if llm is None:
        return None

    prompt = build_prompt(query, context_chunks, sources)
    from langchain.schema import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content
