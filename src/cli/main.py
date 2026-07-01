import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from src.agent.agent import MaritimeLawAgent, get_agent
from src.agent.memory import get_memory
from src.agent.formatter import format_disclaimer
from src.vector_store.store import get_store
from src.config import TOP_K, RERANK_ENABLED

console = Console()


def display_index_stats():
    store = get_store()
    stats = store.get_index_stats()
    console.print(Panel.fit(
        f"[bold]知识库状态[/bold]\n"
        f"  Parent chunks: {stats['parent_chunks']}\n"
        f"  Child chunks: {stats['child_chunks']}\n"
        f"  ChromaDB entries: {stats['chroma_entries']}\n"
        f"  BM25 corpus: {stats['bm25_corpus_size']}\n"
        f"  来源数量: {len(stats['sources'])}\n"
        f"  来源: {', '.join(stats['sources'])}",
        border_style="green",
        title="索引统计",
    ))


def interactive_mode():
    try:
        store = get_store()
    except Exception as e:
        console.print(f"[red]存储初始化失败: {e}[/red]")
        console.print("请先执行 `python scripts/build_knowledge_base.py` 构建知识库。")
        sys.exit(1)

    display_index_stats()

    console.print(Panel.fit(
        "[bold blue]船舶法律法规智能 Agent (v3)[/bold blue]\n"
        "ReACT 循环 | 多工具 | 多轮对话记忆 | 引用验证 | 错误降级\n"
        "命令: agent, direct, history, tools, help, quit, stats, clear",
        border_style="blue",
    ))
    console.print("[dim]提示: 设置 OPENAI_API_KEY 或配置 OLLAMA 启用 LLM 生成回答[/dim]\n")

    agent_mode = True

    while True:
        try:
            query = console.input("[bold green]船舶法> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "help":
            console.print("[yellow]命令:[/yellow]")
            console.print("  agent       - 切换到 Agent 模式 (默认)")
            console.print("  direct      - 切换到直接检索模式")
            console.print("  history     - 显示对话历史")
            console.print("  tools       - 列出可用工具")
            console.print("  stats       - 显示知识库状态")
            console.print("  clear       - 清除对话历史")
            console.print("  quit/exit   - 退出")
            console.print("\n在 Agent 模式下, 支持:")
            console.print("  - 自然语言法律查询")
            console.print("  - 条款精确查找 (如 '海商法第一百七十五条')")
            console.print("  - 法律比较 (如 '比较海商法和SOLAS关于船舶安全的规定')")
            console.print("  - 责任限额计算 (如 '5000总吨船舶的赔偿限额')")
            console.print("  - 多轮追问跟踪")
            continue
        if query.lower() == "agent":
            agent_mode = True
            console.print("[yellow]切换到 Agent 模式 (ReACT 循环)[/yellow]")
            continue
        if query.lower() == "direct":
            agent_mode = False
            console.print("[yellow]切换到直接检索模式[/yellow]")
            continue
        if query.lower() == "history":
            mem = get_memory()
            hist = mem.get_history()
            if not hist:
                console.print("[yellow]暂无对话历史[/yellow]")
            else:
                table = Table(title="对话历史", border_style="blue")
                table.add_column("#", style="dim")
                table.add_column("角色", style="cyan")
                table.add_column("内容摘要", style="white")
                for i, turn in enumerate(hist[-20:], 1):
                    content = turn["content"][:100].replace("\n", " ")
                    table.add_row(str(i), turn["role"], content)
                console.print(table)
            continue
        if query.lower() == "tools":
            from src.agent.tools import TOOL_DESCRIPTIONS
            table = Table(title="可用工具", border_style="blue")
            table.add_column("工具名", style="cyan")
            table.add_column("描述", style="white")
            table.add_column("参数", style="green")
            for name, info in TOOL_DESCRIPTIONS.items():
                params = ", ".join(f"{k}={v}" for k, v in info["params"].items())
                table.add_row(name, info["description"], params)
            console.print(table)
            continue
        if query.lower() == "stats":
            display_index_stats()
            continue
        if query.lower() == "clear":
            get_memory().clear()
            console.print("[yellow]对话历史已清除[/yellow]")
            continue

        try:
            if agent_mode:
                agent = get_agent()
                console.print("[dim]Agent ReACT 模式...[/dim]")
                response = agent.run(query)
                if "未配置LLM" in response or "检索到" in response:
                    console.print(response)
                else:
                    console.print(Panel(
                        response,
                        title="Agent 响应",
                        border_style="green",
                    ))
                console.print(Panel(
                    format_disclaimer(),
                    border_style="dim",
                    title="免责声明",
                ))
            else:
                from src.rag.retriever import MaritimeLawRetriever
                from src.rag.generator import generate_answer
                retriever = MaritimeLawRetriever(top_k=TOP_K, rerank=RERANK_ENABLED)
                results = retriever.retrieve(query)
                context_texts = [doc.page_content for doc, _ in results]
                sources = [doc.metadata.get("source", "unknown") for doc, _ in results]

                table = Table(title="检索结果", border_style="blue")
                table.add_column("#", style="dim")
                table.add_column("来源", style="cyan")
                table.add_column("相关度", style="green", width=10)
                table.add_column("内容", style="white")
                for i, (doc, score) in enumerate(results, 1):
                    src = doc.metadata.get("source", "unknown")
                    snippet = doc.page_content[:100].replace("\n", " ") + "..."
                    table.add_row(str(i), src, f"{score:.4f}", snippet)
                console.print(table)

                answer = generate_answer(query, context_texts, sources)
                if answer:
                    console.print(Panel(Markdown(answer), title="AI回答", border_style="green"))
                else:
                    console.print("[yellow]未配置LLM，显示检索原文:[/yellow]")
                    for i, (doc, score) in enumerate(results, 1):
                        src = doc.metadata.get("source", "unknown")
                        console.print(Panel(
                            doc.page_content,
                            title=f"[{i}] {src} (相关度: {score:.4f})",
                            border_style="cyan",
                        ))

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        agent = get_agent()
        response = agent.run(query)
        console.print(Panel(response, title="Agent Response", border_style="green"))
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
