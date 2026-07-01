#!/usr/bin/env python3
"""Agent system tests."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.memory import SessionMemory, get_memory
from src.agent.planner import classify_query, decompose_complex
from src.agent.tools import retrieve, get_article, compare, calculate, summarize
from src.agent.citation import extract_citations, verify_citations
from src.agent.formatter import format_results, format_comparison, format_calculation
from src.agent.agent import MaritimeLawAgent, get_agent


def test_memory():
    print("\n" + "=" * 60)
    print("[TEST] Conversation Memory")
    print("=" * 60)
    mem = SessionMemory(session_id="test_memory")
    mem.clear()
    mem.add_user("船舶碰撞的法律规定是什么")
    mem.add_assistant("根据《海商法》第八章...")
    assert len(mem.stm) == 2
    assert mem.is_follow_up("具体是怎么规定的")
    assert not mem.is_follow_up("SOLAS公约是什么")
    assert "用户" in mem.get_context_text()
    print("  Follow-up detection: OK")
    print("  History tracking: OK")
    print("  Entities:", mem.info()["entities"])
    print("[PASS] Memory OK")


def test_query_planner():
    print("\n" + "=" * 60)
    print("[TEST] Query Planner")
    print("=" * 60)
    test_cases = [
        ("船舶碰撞的法律规定", "general_retrieve"),
        ("海商法第一百七十五条", "article_lookup"),
        ("比较海商法和SOLAS的安全规定", "comparison"),
        ("5000总吨船舶的赔偿限额", "calculation"),
        ("海商法是什么", "summarize"),
        ("具体怎么赔偿", "follow_up"),
    ]
    for query, expected in test_cases:
        plan = classify_query(query)
        status = "OK" if plan.intent == expected else f"MISMATCH (got {plan.intent})"
        print(f"  [{status}] {query[:40]:40s} -> {plan.intent} (expected {expected})")

    clauses = decompose_complex("船舶碰撞的归责原则；海难救助的报酬计算")
    assert len(clauses) == 2
    print(f"  Decomposition: {clauses}")
    print("[PASS] Query Planner OK")


def test_tools():
    print("\n" + "=" * 60)
    print("[TEST] Agent Tools")
    print("=" * 60)
    results = retrieve("船舶碰撞", k=3)
    assert len(results) > 0
    print(f"  retrieve(): {len(results)} results")

    articles = get_article("海商法", "第一百七十四条")
    print(f"  get_article(): {len(articles)} results")

    comp = compare("海商法,SOLAS", "船舶安全")
    assert "海商法" in comp
    print(f"  compare(): {len(comp)} laws compared")

    calc = calculate(5000, "property")
    assert calc is not None
    print(f"  calculate(): limit={calc['limit_sdr']:.0f} SDR")

    summary = summarize("海商法")
    assert summary is not None
    print(f"  summarize(): {len(summary)} chars")
    print("[PASS] Tools OK")


def test_citation():
    print("\n" + "=" * 60)
    print("[TEST] Citation Verification")
    print("=" * 60)
    text = "根据《海商法》第一百七十五条和《海上交通安全法》的相关规定"
    citations = extract_citations(text)
    assert len(citations) >= 2
    print(f"  Extracted {len(citations)} citations: {[c['full'] for c in citations]}")

    context = []
    from langchain.schema import Document
    context.append(Document(
        page_content="第一百七十五条　船舶发生碰撞",
        metadata={"source": "海商法"},
    ))
    verification = verify_citations(citations, context)
    for cit, status in verification.items():
        print(f"  {cit}: {status}")
    print("[PASS] Citation OK")


def test_agent_pipeline():
    print("\n" + "=" * 60)
    print("[TEST] Agent Pipeline (ReACT)")
    print("=" * 60)
    agent = get_agent()
    test_queries = [
        "船舶碰撞的法律规定是什么",
        "海商法第一百七十五条",
        "比较海商法和SOLAS对船舶安全的要求",
    ]
    for q in test_queries:
        response = agent.run(q)
        assert len(response) > 0
        assert "[ReACT]" in response
        assert "[Think]" in response or "[Plan]" in response
        src_count = response.count("[Act]") + response.count("[Observe]")
        print(f"  [OK] {q[:40]:40s} -> {src_count} ReACT steps, {len(response)} chars")
    print("[PASS] Agent pipeline OK")


def test_error_recovery():
    print("\n" + "=" * 60)
    print("[TEST] Error Recovery & Degradation")
    print("=" * 60)
    from src.agent.planner import QueryPlan
    agent = MaritimeLawAgent()
    response = agent.run("a" * 2000)
    assert len(response) > 0
    print(f"  Long query handled: {len(response)} chars")
    print("[PASS] Error recovery OK")


def test_multi_turn():
    print("\n" + "=" * 60)
    print("[TEST] Multi-turn Conversation")
    print("=" * 60)
    agent = MaritimeLawAgent()
    r1 = agent.run("船舶碰撞的法律规定")
    assert len(r1) > 0
    r2 = agent.run("具体赔偿比例是多少")
    assert len(r2) > 0
    print(f"  Turn 1: {len(r1)} chars")
    print(f"  Turn 2 (follow-up): {len(r2)} chars")
    assert len(agent.memory.stm) >= 2
    print("[PASS] Multi-turn OK")


def test_formatting():
    print("\n" + "=" * 60)
    print("[TEST] Output Formatting")
    print("=" * 60)
    result = format_calculation({"tonnage": 5000, "loss_type": "财产损失", "limit_sdr": 100000, "limit_approx_cny": 950000, "note": "test"})
    assert "SDR" in result
    assert "人民币" in result
    print(f"  calc format: {len(result)} chars")

    import re
    from langchain.schema import Document
    text = "根据《海商法》第一百七十五条"
    from src.agent.citation import CITATION_PATTERN
    assert CITATION_PATTERN.search(text)
    print("  citation regex: OK")
    print("[PASS] Formatting OK")


def main():
    print("=" * 60)
    print("Maritime Law RAG - Agent System Tests")
    print("=" * 60)

    test_memory()
    test_query_planner()
    test_tools()
    test_citation()
    test_agent_pipeline()
    test_error_recovery()
    test_multi_turn()
    test_formatting()

    print("\n" + "=" * 60)
    print("ALL AGENT TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
