from typing import List, Dict, Optional


class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def add_user(self, content: str):
        self.add_turn("user", content)

    def add_assistant(self, content: str):
        self.add_turn("assistant", content)

    def add_system(self, content: str):
        self.add_turn("system", content)

    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        if last_n:
            return self.history[-last_n:]
        return self.history

    def get_context_text(self, last_n: int = 3) -> str:
        recent = self.history[-(last_n * 2):]
        lines = []
        for turn in recent:
            label = {"user": "用户", "assistant": "助手", "system": "系统"}.get(turn["role"], turn["role"])
            content = turn["content"][:200]
            lines.append(f"[{label}]: {content}")
        return "\n".join(lines)

    def is_follow_up(self, query: str) -> bool:
        if not self.history:
            return False
        follow_up_markers = [
            "它", "他", "她", "那", "这", "上述", "以上",
            "具体", "详细", "进一步", "还有", "另外",
            "第", "条", "款", "章", "节",
        ]
        return any(query.startswith(m) or query.startswith(m) for m in follow_up_markers)

    def clear(self):
        self.history.clear()


_memory = None


def get_memory():
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory
