import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = PROCESSED_DIR / "chroma_db"
EVAL_DIR = DATA_DIR / "eval"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("OPENROUTER_API_KEY", "sk-your-api-key-here"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek/deepseek-v4-flash")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:7b")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "1024"))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "128"))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "256"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "32"))

TOP_K = int(os.getenv("TOP_K", "5"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
RRF_K = int(os.getenv("RRF_K", "60"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.5"))
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "BAAI/bge-reranker-v2-m3",
)
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

HNSW_M = int(os.getenv("HNSW_M", "16"))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "50"))

CACHE_MAXSIZE = int(os.getenv("CACHE_MAXSIZE", "128"))

RETRY_SYNONYMS = os.getenv("RETRY_SYNONYMS", "船舶:船,碰撞:撞,救助:救援,赔偿:补偿,责任:义务").split(",")

SOURCES = {
    "maritime_code_wikisource": "https://zh.wikisource.org/wiki/%E4%B8%AD%E5%8D%8E%E4%BA%BA%E6%B0%91%E5%85%B1%E5%92%8C%E5%9B%BD%E6%B5%B7%E5%95%86%E6%B3%95_(2025%E5%B9%B4)",
    "maritime_traffic_safety_law": "https://www.mfa.gov.cn/web/wjb_673085/zzjg_673183/bjhysws_674671/bhflfg/hyfxzhxfl/202303/P020230313589856410683.pdf",
    "msa_regulations": "https://www.msa.gov.cn/msacncms_hsfg/",
}

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
