from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP

SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]


def chunk_text(text: str, source: str = "") -> Tuple[List[dict], List[dict]]:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    parent_texts = parent_splitter.split_text(text)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )

    parent_chunks = []
    child_chunks = []
    for parent_id, parent_text in enumerate(parent_texts):
        parent_chunks.append({
            "text": parent_text,
            "source": source,
            "parent_id": parent_id,
        })
        child_texts = child_splitter.split_text(parent_text)
        for child_id, child_text in enumerate(child_texts):
            child_chunks.append({
                "text": child_text,
                "source": source,
                "child_id": child_id,
                "parent_id": parent_id,
            })

    return parent_chunks, child_chunks


def chunk_all(documents: dict) -> Tuple[List[dict], List[dict]]:
    all_parents = []
    all_children = []
    global_parent_id = 0

    for source_name, text in documents.items():
        local_parents, local_children = chunk_text(text, source_name)
        pid_map = {}
        for p in local_parents:
            old_pid = p["parent_id"]
            p["parent_id"] = global_parent_id
            pid_map[old_pid] = global_parent_id
            all_parents.append(p)
            global_parent_id += 1
        for c in local_children:
            c["parent_id"] = pid_map[c["parent_id"]]
            all_children.append(c)
        print(f"[chunker] {source_name} -> {len(local_parents)} parents, {len(local_children)} children")

    return all_parents, all_children
