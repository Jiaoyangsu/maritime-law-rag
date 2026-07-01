import json
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, PROCESSED_DIR

MEMORY_DIR = PROCESSED_DIR / "sessions"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

STM_MAX_EXCHANGES = 10
LTM_SUMMARY_INTERVAL = 5

ENTITY_PATTERNS = {
    "laws": re.compile(r"[《]([^》]+?(?:法|条例|公约|规则))[》]"),
    "articles": re.compile(r"第\s*([一二三四五六七八九十百千\d]+)\s*(条|章|节)"),
    "concepts": re.compile(r"(碰撞|救助|赔偿|责任|保险|优先权|共同海损|时效|限额|污染|排放|登记|证书|吨税|航行|港口|船员|托运|承运)"),
}

LONG_TERM_SUMMARY_PROMPT = """你是一个法律对话记录员。请将以下多轮法律对话压缩为结构化的长期记忆摘要，保留关键信息。

请按以下格式输出：
- 涉及法律: 列出用户询问过的主要法律/公约名称
- 关键条款: 列出被引用或讨论的具体条款编号
- 核心问题: 列出用户提出的主要问题及其结论
- 待办事项: 如果用户要求后续跟进某些内容，列出来

对话记录：
{history}"""


@dataclass
class MemoryExchange:
    role: str
    content: str
    timestamp: str = ""
    entities: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.entities:
            self.entities = _extract_entities(self.content)


@dataclass
class LongTermSummary:
    timestamp: str = ""
    laws: List[str] = field(default_factory=list)
    articles: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)
    raw: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def format(self) -> str:
        parts = []
        if self.laws:
            parts.append(f"涉及法律: {'、'.join(self.laws)}")
        if self.articles:
            parts.append(f"关键条款: {', '.join(self.articles)}")
        if self.questions:
            parts.append(f"核心问题: {'; '.join(self.questions)}")
        if self.conclusions:
            parts.append(f"结论: {'; '.join(self.conclusions)}")
        if self.pending:
            parts.append(f"待办: {'; '.join(self.pending)}")
        return "\n".join(parts)


def _extract_entities(text: str) -> Dict[str, List[str]]:
    entities = {}
    for key, pattern in ENTITY_PATTERNS.items():
        matches = pattern.findall(text)
        if key == "articles":
            entities[key] = [f"{n}{s}" for n, s in matches]
        else:
            entities[key] = list(set(matches))
    return entities


def _merge_entities(a: Dict[str, List[str]], b: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged = {}
    all_keys = set(a.keys()) | set(b.keys())
    for k in all_keys:
        seen = set()
        items = []
        for item in a.get(k, []) + b.get(k, []):
            if item not in seen:
                seen.add(item)
                items.append(item)
        merged[k] = items
    return merged


class MemoryStorage:
    def __init__(self, session_dir: Path):
        self.path = session_dir / "memory.json"
        self.summaries_path = session_dir / "summaries.json"

    def save(self, stm: List[dict], ltm: List[dict], entities: dict):
        data = {
            "stm": stm,
            "entities": {k: list(v) if isinstance(v, set) else v for k, v in entities.items()},
        }
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        if ltm:
            self.summaries_path.write_text(json.dumps(ltm, ensure_ascii=False, indent=2))

    def load(self) -> Tuple[List[dict], List[dict], dict]:
        stm = []
        ltm = []
        entities = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            stm = data.get("stm", [])
            entities = data.get("entities", {})
        if self.summaries_path.exists():
            ltm = json.loads(self.summaries_path.read_text())
        return stm, ltm, entities


class SessionMemory:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.session_dir = MEMORY_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.stm: List[MemoryExchange] = []
        self.ltm: List[LongTermSummary] = []
        self.entities: Dict[str, List[str]] = {}
        self._exchange_since_summary = 0

        self._storage = MemoryStorage(self.session_dir)
        self._load()

    def _load(self):
        stm_data, ltm_data, entities = self._storage.load()
        self.stm = [MemoryExchange(**e) for e in stm_data]
        self.ltm = [LongTermSummary(**e) for e in ltm_data]
        self.entities = entities
        self._exchange_since_summary = len(self.stm)

    def _save(self):
        self._storage.save(
            [asdict(e) for e in self.stm],
            [asdict(e) for e in self.ltm],
            self.entities,
        )

    def add_exchange(self, role: str, content: str):
        exchange = MemoryExchange(role=role, content=content)
        self.stm.append(exchange)
        self.entities = _merge_entities(self.entities, exchange.entities)
        self._exchange_since_summary += 1

        if len(self.stm) > STM_MAX_EXCHANGES:
            self.stm = self.stm[-STM_MAX_EXCHANGES:]

        if self._exchange_since_summary >= LTM_SUMMARY_INTERVAL:
            self._summarize()

        self._save()

    def add_user(self, content: str):
        self.add_exchange("user", content)

    def add_assistant(self, content: str):
        self.add_exchange("assistant", content)

    def _summarize(self):
        if not OPENAI_API_KEY:
            return
        history_text = []
        for e in self.stm[:self._exchange_since_summary]:
            label = "用户" if e.role == "user" else "助手"
            history_text.append(f"[{label}]: {e.content[:300]}")
        if not history_text:
            return

        try:
            from langchain_openai import ChatOpenAI
            from langchain.schema import SystemMessage, HumanMessage

            kwargs = {"model": OPENAI_MODEL, "api_key": OPENAI_API_KEY, "temperature": 0, "max_tokens": 500}
            if OPENAI_BASE_URL:
                kwargs["base_url"] = OPENAI_BASE_URL
            llm = ChatOpenAI(**kwargs)

            resp = llm.invoke([
                SystemMessage(content="你是一个法律对话记录员，只输出摘要。"),
                HumanMessage(content=LONG_TERM_SUMMARY_PROMPT.format(history="\n".join(history_text))),
            ])
            raw = resp.content.strip()

            summary = LongTermSummary(raw=raw)
            for key, label in [("laws", "涉及法律"), ("articles", "关键条款"), ("concepts", "核心问题"),
                               ("pending", "待办")]:
                for line in raw.split("\n"):
                    if label in line:
                        val = line.split(":", 1)[1].strip()
                        setattr(summary, key, [v.strip() for v in val.split("、") if v.strip()])
            self.ltm.append(summary)
            self._exchange_since_summary = 0
        except Exception:
            pass

    def get_context_text(self, last_n: int = 3) -> str:
        recent = self.stm[-(last_n * 2):]
        lines = []
        for e in recent:
            label = "用户" if e.role == "user" else "助手"
            lines.append(f"[{label}]: {e.content[:200]}")
        return "\n".join(lines)

    def get_entity_context(self) -> str:
        parts = []
        for key, label in [("laws", "已涉及法律"), ("articles", "已提及条款"), ("concepts", "核心概念")]:
            items = self.entities.get(key, [])
            if items:
                parts.append(f"{label}: {'、'.join(items[:10])}")
        return "\n".join(parts)

    def get_ltm_context(self) -> str:
        if not self.ltm:
            return ""
        lines = ["[长期记忆]"]
        for s in self.ltm[-3:]:
            lines.append(s.format())
        return "\n".join(lines)

    def is_follow_up(self, query: str) -> bool:
        if not self.stm:
            return False
        follow_up_markers = [
            "它", "他", "她", "那", "这", "上述", "以上",
            "具体", "详细", "进一步", "还有", "另外",
            "第", "条", "款", "章", "节",
        ]
        return any(query.startswith(m) for m in follow_up_markers)

    def clear(self):
        self.stm.clear()
        self.ltm.clear()
        self.entities.clear()
        self._exchange_since_summary = 0
        if self.path.exists():
            self.path.unlink()
        if self.summaries_path.exists():
            self.summaries_path.unlink()
        self._save()

    def info(self) -> dict:
        return {
            "session_id": self.session_id,
            "stm_exchanges": len(self.stm),
            "ltm_summaries": len(self.ltm),
            "entities": {k: len(v) for k, v in self.entities.items()},
        }

    @property
    def path(self) -> Path:
        return self._storage.path

    @property
    def summaries_path(self) -> Path:
        return self._storage.summaries_path


_active_sessions: Dict[str, SessionMemory] = {}


def get_memory(session_id: Optional[str] = None) -> SessionMemory:
    sid = session_id or "default"
    if sid not in _active_sessions:
        _active_sessions[sid] = SessionMemory(session_id=sid)
    return _active_sessions[sid]


def list_sessions() -> List[str]:
    return [d.name for d in MEMORY_DIR.iterdir() if d.is_dir() and (d / "memory.json").exists()]
