#!/usr/bin/env python3
"""OpenAI-compatible API server for Open WebUI integration."""
import os
import sys
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from src.rag.retriever import MaritimeLawRetriever
from src.rag.generator import generate_answer, build_prompt, SYSTEM_PROMPT
from src.config import TOP_K, RERANK_ENABLED, OPENAI_API_KEY, LLM_PROVIDER
from src.vector_store.store import get_store

app = FastAPI(title="Maritime Law RAG API")

retriever = MaritimeLawRetriever(top_k=TOP_K, rerank=RERANK_ENABLED)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "maritime-law-rag"
    messages: List[Message]
    stream: bool = False
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None


def build_response(content: str, model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        return JSONResponse({"error": "No messages provided"}, status_code=400)

    query = req.messages[-1].content or ""

    results = retriever.retrieve(query)
    context_chunks = [doc.page_content for doc, _ in results]
    sources = [doc.metadata.get("source", "未知") for doc, _ in results]

    memory_context = None
    if len(req.messages) > 1:
        history = []
        for m in req.messages[-6:-1]:
            history.append(f"{'用户' if m.role == 'user' else '助手'}: {m.content}")
        memory_context = "\n".join(history)

    if OPENAI_API_KEY and "sk-" in OPENAI_API_KEY:
        try:
            answer = generate_answer(query, context_chunks, sources, memory_context)
        except Exception:
            answer = None
    else:
        answer = None

    if answer is None:
        context = "\n\n---\n\n".join(
            f"[来源: {src}]\n{chunk}"
            for chunk, src in zip(context_chunks, sources)
        )
        answer = f"根据检索到的法条：\n\n{context}"

    seen = set()
    citations = [src for src in sources if not (src in seen or seen.add(src))]
    answer += "\n\n---\n参考来源：" + "\n".join(f"- {s}" for s in citations)

    if req.stream:
        content = json.dumps(build_response(answer, req.model), ensure_ascii=False)

        async def stream():
            yield f"data: {json.dumps(build_response(answer, req.model), ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return JSONResponse(build_response(answer, req.model))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "maritime-law-rag",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "self",
            }
        ],
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"[API] Maritime Law RAG server starting on http://0.0.0.0:{port}")
    print(f"[API] Connect Open WebUI to: http://localhost:{port}/v1")
    uvicorn.run(app, host="0.0.0.0", port=port)
