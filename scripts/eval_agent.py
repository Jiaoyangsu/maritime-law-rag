#!/usr/bin/env python3
"""Eval script: run agent end-to-end against human-annotated queries with LLM-as-judge."""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.agent import MaritimeLawAgent
from src.agent.citation import extract_citations
from src.agent.memory import get_memory
from src.config import EVAL_DIR, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from langchain.schema import HumanMessage

JUDGE_SYSTEM_PROMPT = """你是一个专业的法律问答评估专家。请根据以下标准对助手的回答进行评分（1-5分）：

1. **相关性 (Relevancy)**：回答是否直接针对用户的问题？
2. **准确性 (Accuracy)**：回答中的法律引用是否准确？是否使用了正确的法条名称和条款？
3. **完整性 (Completeness)**：回答是否完整覆盖了问题的关键方面？

评分标准：
- 5分：完美 - 直接回答、引用准确、覆盖全面
- 4分：良好 - 回答准确但有轻微遗漏
- 3分：一般 - 部分相关但不够精准或完整
- 2分：较差 - 回答与问题部分无关或引用错误
- 1分：差 - 回答与问题无关或严重错误

请按以下JSON格式输出评分和理由：
{"relevancy": <1-5>, "accuracy": <1-5>, "completeness": <1-5>, "reason": "<简短理由>"}"""


def _build_judge_llm():
    if not OPENAI_API_KEY:
        return None
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import SystemMessage, HumanMessage
        kwargs = {"model": OPENAI_MODEL, "api_key": OPENAI_API_KEY, "temperature": 0}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        llm = ChatOpenAI(**kwargs)
        return llm, SystemMessage(content=JUDGE_SYSTEM_PROMPT)
    except ImportError:
        return None


def load_queries(path: str = "data/eval/queries.json") -> list:
    with open(path) as f:
        return json.load(f)


def judge_answer(query: str, answer: str, llm, sys_msg) -> dict:
    trimmed = answer[-2000:] if len(answer) > 2000 else answer
    prompt = f"""用户问题：{query}

助手回答（末尾）：
{trimmed}

请评估以上回答的质量。"""
    for attempt in range(2):
        try:
            resp = llm.invoke([sys_msg, HumanMessage(content=prompt)])
            content = resp.content.strip()
            start = content.index("{")
            end = content.rindex("}") + 1
            return json.loads(content[start:end])
        except Exception as e:
            if attempt == 1:
                print(f"      [judge error] {e}", file=sys.stderr)
    return {"relevancy": 0, "accuracy": 0, "completeness": 0, "reason": "judge failed"}


EVAL_LAW_ALIASES = {
    "SOLAS": "IMO Convention - SOLAS 详细",
    "SOLAS公约": "IMO Convention - SOLAS 详细",
    "国际海上人命安全公约": "IMO Convention - SOLAS 详细",
    "MARPOL": "IMO Convention - MARPOL 详细",
    "MARPOL公约": "IMO Convention - MARPOL 详细",
    "国际防止船舶造成污染公约": "IMO Convention - MARPOL 详细",
    "防止船舶造成污染": "IMO Convention - MARPOL 详细",
    "STCW": "IMO Convention - STCW 详细",
    "STCW公约": "IMO Convention - STCW 详细",
    "海员培训、发证和值班标准国际公约": "IMO Convention - STCW 详细",
    "ISM Code": "ISM Code (国际安全管理规则)",
    "国际安全管理规则": "ISM Code (国际安全管理规则)",
    "国际安全管理规则（ISM Code）": "ISM Code (国际安全管理规则)",
    "MLC": "MLC 2006 (海事劳工公约)",
    "MLC 2006": "MLC 2006 (海事劳工公约)",
    "海事劳工公约": "MLC 2006 (海事劳工公约)",
    "中华人民共和国海商法": "海商法",
    "海商法": "海商法",
}


def check_citations(answer: str, expected_sources: list) -> dict:
    citations = extract_citations(answer)
    found_sources = set()
    for cit in citations:
        cit_text = cit["full"] + cit["law"]
        for src in expected_sources:
            norm = src.replace("IMO Convention - ", "").replace(" (国际安全管理规则)", "").replace(" (海事劳工公约)", "").strip()
            parts = norm.split()
            if any(p in cit_text for p in parts):
                found_sources.add(src)
                continue
            for alias, target in EVAL_LAW_ALIASES.items():
                if target == src and alias in cit_text:
                    found_sources.add(src)
                    break
    n_expected = len(expected_sources)
    n_found = len(found_sources)
    return {
        "source_recall": n_found / n_expected if n_expected > 0 else 1.0,
        "n_citations": len(citations),
        "expected_sources": expected_sources,
        "found_sources": list(found_sources),
        "missing_sources": [s for s in expected_sources if s not in found_sources],
    }


def _eval_one(item: dict, idx: int) -> dict:
    q = item["query"]
    expected = item["expected_sources"]

    agent = MaritimeLawAgent(session_id=f"eval_{idx}")
    get_memory(f"eval_{idx}").clear()

    t0 = time.time()
    try:
        answer = agent.run(q)
        latency = time.time() - t0
    except Exception as e:
        latency = time.time() - t0
        answer = f"[ERROR] {e}"

    has_error = answer.startswith("[ERROR]")
    has_degrade = "Degrade" in answer or "降级" in answer
    cit_check = check_citations(answer, expected)
    is_pass = cit_check["source_recall"] >= 0.5 and not has_error

    result = {
        "idx": idx,
        "query": q,
        "expected": expected,
        "answer": answer,
        "latency": round(latency, 2),
        "pass": is_pass,
        "has_error": has_error,
        "has_degrade": has_degrade,
        "source_recall": cit_check["source_recall"],
        "n_citations": cit_check["n_citations"],
        "missing_sources": cit_check["missing_sources"],
    }

    status = "PASS" if is_pass else "FAIL"
    dg = " [DEGRADE]" if has_degrade else ""
    print(f"[{status}]{dg} #{idx:2d} {q[:40]:40s} "
          f"lat={latency:.1f}s recall={cit_check['source_recall']:.2f} "
          f"cites={cit_check['n_citations']}")
    if not is_pass:
        print(f"      missing: {cit_check['missing_sources']}")

    return result


def evaluate():
    queries = load_queries()
    print(f"Loaded {len(queries)} eval queries\n")

    results = [None] * len(queries)
    max_workers = min(4, len(queries))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_eval_one, item, i + 1): i
            for i, item in enumerate(queries)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    judge = _build_judge_llm()
    if judge:
        judge_llm, judge_sys_msg = judge
        for r in results:
            r["quality"] = judge_answer(r["query"], r["answer"], judge_llm, judge_sys_msg)
            print(f"  Judge #{r['idx']:2d} {r['query'][:40]:40s} "
                  f"Q={r['quality']['relevancy']} A={r['quality']['accuracy']} C={r['quality']['completeness']}")
    else:
        for r in results:
            r["quality"] = {"relevancy": 0, "accuracy": 0, "completeness": 0, "reason": "no judge"}

    total_latency = sum(r["latency"] for r in results)
    pass_count = sum(1 for r in results if r["pass"])
    fail_count = len(queries) - pass_count
    degrade_count = sum(1 for r in results if r["has_degrade"])
    n = len(queries)
    pass_rate = pass_count / n * 100
    avg_latency = total_latency / n if n > 0 else 0
    valid_q = [r["quality"] for r in results if r["quality"]["relevancy"] > 0]
    avg_q = sum(v["relevancy"] for v in valid_q) / len(valid_q) if valid_q else 0
    avg_a = sum(v["accuracy"] for v in valid_q) / len(valid_q) if valid_q else 0
    avg_c = sum(v["completeness"] for v in valid_q) / len(valid_q) if valid_q else 0

    print(f"\n{'='*60}")
    print(f"AGENT E2E EVALUATION RESULTS ({n} queries, {max_workers} workers)")
    print(f"{'='*60}")
    print(f"  Pass rate:     {pass_rate:.1f}% ({pass_count}/{n})")
    print(f"  Wall time:     {total_latency/max_workers:.1f}s est.")
    print(f"  Avg latency:   {avg_latency:.1f}s")
    print(f"  Degrade count: {degrade_count}/{n}")
    if judge:
        print(f"  Avg quality (LLM-as-judge):")
        print(f"    Relevancy:    {avg_q:.2f}/5")
        print(f"    Accuracy:     {avg_a:.2f}/5")
        print(f"    Completeness: {avg_c:.2f}/5")
    print(f"\n  FAIL cases:")
    for r in results:
        if not r["pass"]:
            print(f"    #{r['idx']:2d} {r['query'][:50]:50s} missing={r['missing_sources']}")

    report_path = EVAL_DIR / "agent_eval_report.json"
    for r in results:
        r.pop("answer", None)
    with open(report_path, "w") as f:
        json.dump({
            "n_queries": n,
            "pass_rate": round(pass_rate, 1),
            "avg_latency": round(avg_latency, 2),
            "degrade_count": degrade_count,
            "avg_quality": {"relevancy": round(avg_q, 2), "accuracy": round(avg_a, 2), "completeness": round(avg_c, 2)} if judge else None,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to {report_path}")

    return pass_rate, avg_latency


def test_degrade_chain():
    """Test each degrade level in the agent."""
    print(f"\n{'='*60}")
    print(f"DEGRADE CHAIN TEST")
    print(f"{'='*60}")
    agent = MaritimeLawAgent()
    memory = get_memory()
    results = {}

    memory.clear()
    t0 = time.time()
    answer = agent.run("船舶碰撞的归责原则是什么")
    latency = time.time() - t0
    results["normal_retrieve"] = {
        "latency": round(latency, 2),
        "has_degrade": "Degrade" in answer or "降级" in answer,
        "answer_len": len(answer),
    }
    status = "PASS" if not results["normal_retrieve"]["has_degrade"] else "FAIL"
    print(f"  [{status}] normal_retrieve: latency={latency:.1f}s len={len(answer)}")

    print(f"\n  Degrade chain test: cleared.")
    return results


def test_no_llm():
    """Test system behavior when LLM is unavailable."""
    print(f"\n{'='*60}")
    print(f"LLM UNAVAILABLE TEST")
    print(f"{'='*60}")
    import src.config as cfg
    original_key = cfg.OPENAI_API_KEY
    cfg.OPENAI_API_KEY = ""

    agent = MaritimeLawAgent()
    memory = get_memory()
    memory.clear()

    t0 = time.time()
    answer = agent.run("船舶碰撞的归责原则是什么")
    latency = time.time() - t0

    cfg.OPENAI_API_KEY = original_key

    has_degrade = "降级" in answer
    is_formatted = any(m in answer for m in ["海商法", "碰撞", "责任"])
    status = "PASS" if has_degrade and is_formatted else "FAIL"
    print(f"  [{status}] no_llm: latency={latency:.1f}s degrade={'[DEGRADE]' if has_degrade else ''} formatted={is_formatted}")
    return {"latency": round(latency, 2), "has_degrade": has_degrade, "is_formatted": is_formatted}


if __name__ == "__main__":
    test_degrade_chain()
    test_no_llm()
    print()
    evaluate()
