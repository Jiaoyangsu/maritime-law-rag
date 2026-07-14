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

    try:
        answer = generate_answer(query, context_chunks, sources, memory_context)
    except Exception as e:
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


@app.get("/")
async def web_ui():
    return HTMLResponse(HTML_PAGE)


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>海事法律 RAG</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif; background: #f5f5f5; height: 100vh; display: flex; flex-direction: column; }
  header { background: #1a365d; color: white; padding: 16px 24px; font-size: 18px; font-weight: 600; display: flex; align-items: center; gap: 10px; }
  header span { font-size: 12px; color: #90cdf4; font-weight: 400; }
  #chat { flex: 1; overflow-y: auto; padding: 20px; max-width: 800px; margin: 0 auto; width: 100%; }
  .msg { margin-bottom: 16px; display: flex; flex-direction: column; }
  .msg.user { align-items: flex-end; }
  .msg.assistant { align-items: flex-start; }
  .bubble { max-width: 85%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; font-size: 14px; white-space: pre-wrap; }
  .user .bubble { background: #1a365d; color: white; border-bottom-right-radius: 4px; }
  .assistant .bubble { background: white; color: #333; border: 1px solid #e2e8f0; border-bottom-left-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .sources { font-size: 12px; color: #718096; margin-top: 4px; padding-left: 4px; }
  #input-area { border-top: 1px solid #e2e8f0; background: white; padding: 16px 20px; }
  #form { max-width: 800px; margin: 0 auto; display: flex; gap: 8px; }
  #input { flex: 1; padding: 10px 14px; border: 1px solid #e2e8f0; border-radius: 8px; font-size: 14px; outline: none; }
  #input:focus { border-color: #1a365d; }
  #send { padding: 10px 24px; background: #1a365d; color: white; border: none; border-radius: 8px; font-size: 14px; cursor: pointer; }
  #send:hover { background: #2a4a7d; }
  #send:disabled { background: #a0aec0; cursor: not-allowed; }
  .loading { color: #a0aec0; font-size: 13px; padding: 8px 0; }
</style>
</head>
<body>
<header>⚓ 海事法律智能问答 <span>Maritime Law RAG</span></header>
<div id="chat">
  <div class="msg assistant"><div class="bubble">你好！我是海事法律助手，可以查询中国及国际海事法律法规。请问有什么可以帮你的？</div></div>
</div>
<div id="input-area">
  <div id="form">
    <input id="input" type="text" placeholder="输入问题，如：船舶碰撞的法律规定" autofocus>
    <button id="send" onclick="send()">发送</button>
  </div>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');

input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

async function send() {
  const text = input.value.trim();
  if (!text) return;

  addMsg('user', text);
  input.value = '';
  sendBtn.disabled = true;

  const loading = document.createElement('div');
  loading.className = 'msg assistant';
  loading.innerHTML = '<div class="loading">思考中...</div>';
  chat.appendChild(loading);
  chat.scrollTop = chat.scrollHeight;

  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: text }], stream: false })
    });
    const data = await res.json();
    loading.remove();
    addMsg('assistant', data.choices[0].message.content);
  } catch (e) {
    loading.remove();
    addMsg('assistant', '请求失败，请检查服务是否正常运行。');
  }

  sendBtn.disabled = false;
  input.focus();
}

function addMsg(role, content) {
  const [body, ...rest] = content.split('\\n\\n参考来源：');
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  el.innerHTML = `<div class="bubble">${escapeHtml(body)}</div>`;
  if (rest.length) {
    el.innerHTML += `<div class="sources">📎 参考来源：${escapeHtml(rest.join(''))}</div>`;
  }
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    from fastapi.responses import HTMLResponse
    port = int(os.getenv("PORT", "8080"))
    print(f"[API] Maritime Law RAG server: http://localhost:{port}")
    print(f"[API] Compatible with Open WebUI: http://localhost:{port}/v1")
    uvicorn.run(app, host="0.0.0.0", port=port)
