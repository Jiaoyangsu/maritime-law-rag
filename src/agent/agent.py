from typing import List, Tuple, Optional, Dict
from langchain.schema import Document
from src.vector_store.store import get_store
from src.rag.retriever import MaritimeLawRetriever
from src.rag.reranker import get_reranker
from src.rag.generator import build_prompt, generate_answer, SYSTEM_PROMPT
from src.agent.memory import SessionMemory, get_memory
from src.agent.tools import TOOL_FUNCTIONS, TOOL_DESCRIPTIONS
from src.agent.planner import classify_query, decompose_complex, QueryPlan
from src.agent.citation import extract_citations, verify_citations, has_unverified, get_unverified
from src.agent.query_rewriter import rewrite_query
from src.agent.formatter import (
    format_results,
    format_comparison,
    format_calculation,
    format_article,
    format_summary,
    format_agent_step,
    format_disclaimer,
    format_source_refs,
)
from src.config import TOP_K, RERANK_ENABLED


MAX_REACT_CYCLES = 3


class MaritimeLawAgent:
    def __init__(self, session_id: Optional[str] = None):
        self.memory = get_memory(session_id)
        self.retriever = MaritimeLawRetriever(top_k=TOP_K, rerank=RERANK_ENABLED)
        self.store = get_store()
        self.tools = TOOL_FUNCTIONS
        self.cycle_count = 0

    def run(self, user_query: str) -> str:
        self.cycle_count = 0
        output_lines = []
        reasoning_steps = []

        self.memory.add_user(user_query)

        context_text = self.memory.get_context_text(last_n=2) if self.memory.is_follow_up(user_query) else ""
        entity_ctx = self.memory.get_entity_context()
        if entity_ctx:
            context_text = f"{context_text}\n{entity_ctx}" if context_text else entity_ctx
        rewritten = rewrite_query(user_query, context=context_text)
        if rewritten != user_query:
            output_lines.append(format_agent_step("Rewrite", f"原始: \"{user_query}\" → 改写: \"{rewritten}\""))

        output_lines.append(format_agent_step("ReACT", "开始处理查询"))
        output_lines.append(format_agent_step("Think", f"理解用户意图: \"{user_query}\""))

        plan = classify_query(user_query)
        output_lines.append(format_agent_step("Plan", f"意图={plan.intent}, 工具={[t['name'] for t in plan.tools]}"))
        reasoning_steps.append(f"classified as: {plan.intent}")

        if self.memory.is_follow_up(user_query):
            lines = [self.memory.get_context_text(last_n=2)]
            ltm = self.memory.get_ltm_context()
            if ltm:
                lines.append(ltm)
            context = "\n".join(lines)
            output_lines.append(format_agent_step("Think", f"检测到追问, 引用上文: {context[:100]}"))
            reasoning_steps.append("follow-up detected, using conversation context")

        final_context: List[Document] = []
        final_sources: List[str] = []
        tool_outputs: List[str] = []

        for tool_call in plan.tools:
            result = self._execute_tool(tool_call, output_lines, reasoning_steps)
            if isinstance(result, list):
                for doc, score in result:
                    if doc not in final_context:
                        final_context.append(doc)
                        final_sources.append(doc.metadata.get("source", "unknown"))
            elif isinstance(result, str) and result.strip():
                tool_outputs.append(result)

        if tool_outputs:
            answer = "\n\n".join(tool_outputs)
            output_lines.append("")
            output_lines.append(answer)
            output_lines.append("")
            output_lines.append(format_source_refs(final_context))
            output_lines.append(format_disclaimer())
            self.memory.add_assistant(answer)
            return self._wrap_output(output_lines)

        if final_context:
            output_lines.append(format_agent_step("Observe", f"检索到 {len(final_context)} 条相关段落, 来自 {len(set(final_sources))} 部法律"))

            reranker = get_reranker()
            if reranker and RERANK_ENABLED and len(final_context) > 1:
                output_lines.append(format_agent_step("Act", "Cross-Encoder 重排序中..."))
                try:
                    reranked = reranker.rerank(
                        user_query,
                        [(doc, 1.0) for doc in final_context],
                    )
                    final_context = [doc for doc, _ in reranked]
                    output_lines.append(format_agent_step("Observe", "重排序完成"))
                except Exception as e:
                    output_lines.append(format_agent_step("Degrade", f"Cross-Encoder 不可用 ({e}), 跳过重排序"))

            memory_ctx = self.memory.get_ltm_context()
            entity_ctx = self.memory.get_entity_context()
            combined_memory = "\n".join(filter(None, [memory_ctx, entity_ctx]))

            answer = generate_answer(
                user_query,
                [d.page_content for d in final_context],
                final_sources,
                memory_context=combined_memory or None,
            )
            if answer:
                output_lines.append(format_agent_step("Think", "LLM 生成回答完成"))
                citations = extract_citations(answer)
                if citations:
                    output_lines.append(format_agent_step("Verify", f"检测到 {len(citations)} 处引用，进行验证..."))
                    verification = verify_citations(citations, final_context)
                    unverified = get_unverified(citations, final_context)
                    for cit in unverified:
                        output_lines.append(format_agent_step("Warning", f"引用 \"{cit}\" 未在检索结果中找到对应条款"))
                output_lines.append("")
                output_lines.append(answer)
                output_lines.append("")
                output_lines.append(format_source_refs(final_context))
                output_lines.append(format_disclaimer())
                self.memory.add_assistant(answer)
                return self._wrap_output(output_lines)

            output_lines.append(format_agent_step("Degrade", "LLM 不可用, 降级为规则格式化输出"))
            output_lines.append("")
            output_lines.append(format_results(list(zip(final_context, [1.0] * len(final_context)))))
            output_lines.append("")
            output_lines.append(format_disclaimer())
            self.memory.add_assistant(output_lines[-2])
            return self._wrap_output(output_lines)

        output_lines.append(format_agent_step("Observe", "第一轮检索无结果"))
        recovered = self._recover_retrieve(user_query, output_lines, reasoning_steps)
        if recovered:
            final_context = recovered
            output_lines.append(format_agent_step("Degrade", f"同义扩展检索到 {len(final_context)} 条结果"))
            output_lines.append("")
            output_lines.append(format_results(list(zip(final_context, [1.0] * len(final_context)))))
            output_lines.append("")
            output_lines.append(format_disclaimer())
            self.memory.add_assistant(output_lines[-2])
            return self._wrap_output(output_lines)

        output_lines.append(format_agent_step("Degrade", "所有检索策略均无结果"))
        output_lines.append("")
        output_lines.append("抱歉，经过多次尝试后仍未能检索到与您问题相关的法律条文。请尝试：\n"
                           "1. 使用更简洁的表述\n"
                           "2. 更换关键词\n"
                           "3. 提供更多具体细节\n"
                           "4. 在命令提示符输入 'tools' 查看可用的检索工具")
        output_lines.append("")
        output_lines.append(format_disclaimer())
        self.memory.add_assistant(output_lines[-2])
        return self._wrap_output(output_lines)

    def _recover_retrieve(self, query: str, output_lines: list, reasoning: list) -> Optional[List[Document]]:
        output_lines.append(format_agent_step("Recover", "策略1: 简化查询重试"))
        simple = query[:80]
        results = self.store.hybrid_search(simple, k=TOP_K)
        if results:
            return [doc for doc, _ in results]

        from src.agent.tools import RETRY_SYNONYMS_FALLBACK
        output_lines.append(format_agent_step("Recover", "策略2: 同义词扩展重试"))
        for pair in RETRY_SYNONYMS_FALLBACK:
            if ":" not in pair:
                continue
            a, b = pair.split(":", 1)
            if a in query:
                expanded = query.replace(a, b)
                results = self.store.hybrid_search(expanded, k=TOP_K)
                if results:
                    return [doc for doc, _ in results]

        output_lines.append(format_agent_step("Recover", "策略3: 拆词检索"))
        words = [w for w in query if '\u4e00' <= w <= '\u9fff']
        for length in [4, 3, 2]:
            if len(words) >= length:
                short = "".join(words[:length])
                results = self.store.hybrid_search(short, k=TOP_K)
                if results:
                    return [doc for doc, _ in results]

        return None

    def _execute_tool(self, tool_call: dict, output_lines: list, reasoning: list):
        name = tool_call["name"]
        params = tool_call["params"]
        output_lines.append(format_agent_step("Act", f"执行工具: {name}(params={params})"))
        reasoning.append(f"executed {name}")

        try:
            tool_fn = self.tools[name]
        except KeyError:
            output_lines.append(format_agent_step("Error", f"未知工具: {name}"))
            return []

        try:
            result = tool_fn(**params)
        except Exception as e:
            output_lines.append(format_agent_step("Error", f"工具执行失败: {e}"))

            if name == "retrieve":
                output_lines.append(format_agent_step("Degrade", "检索工具异常, 降级: 简化查询重试"))
                simple_query = params.get("query", "")[:50]
                try:
                    result = TOOL_FUNCTIONS["retrieve"](query=simple_query, k=3)
                    if result:
                        output_lines.append(format_agent_step("Degrade", "降级检索成功"))
                        return result
                except Exception:
                    pass
            elif name == "get_article":
                output_lines.append(format_agent_step("Degrade", "精确条款查找异常, 降级: 语义检索"))
                law = params.get("law", "")
                article = params.get("article", "")
                try:
                    result = TOOL_FUNCTIONS["retrieve"](query=f"{law} {article}", k=5)
                    if result:
                        output_lines.append(format_agent_step("Degrade", "降级检索成功"))
                        return result
                except Exception:
                    pass
            return []

        output_lines.append(format_agent_step("Observe", "工具返回结果"))

        if name == "calculate":
            return format_calculation(result)
        if name == "compare":
            return format_comparison(result)
        if name == "get_article":
            if result:
                return format_article(result)
            output_lines.append(format_agent_step("Degrade", "精确查找无结果, 降级为语义检索"))
            try:
                fallback = TOOL_FUNCTIONS["retrieve"](query=f"{params.get('law', '')} {params.get('article', '')}", k=5)
                if fallback:
                    output_lines.append(format_agent_step("Degrade", "降级检索成功"))
                    return fallback
            except Exception:
                pass
            return []
        if name == "summarize":
            if result:
                return format_summary(result)
            return "未找到该法律的相关概述。"

        return result

    @staticmethod
    def _wrap_output(lines: list) -> str:
        return "\n".join(lines)


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = MaritimeLawAgent()
    return _agent
