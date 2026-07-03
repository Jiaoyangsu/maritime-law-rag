# 船舶法律法规智能问答系统 (Maritime Law RAG Agent)

基于 RAG (Retrieval-Augmented Generation) 的船舶法律法规智能问答系统。

## 已收录法律法规 (15部)

### 中国法律 (7部)
| 法律名称 | 状态 |
|---------|------|
| 《中华人民共和国海商法》(2025修订) | ✅ |
| 《中华人民共和国海上交通安全法》(2021修订) | ✅ |
| 《中华人民共和国海洋环境保护法》(2023修订) | ✅ |
| 《中华人民共和国船舶登记条例》 | ✅ |
| 《中华人民共和国船员条例》 | ✅ |
| 《中华人民共和国船舶吨税法》 | ✅ |
| 《中华人民共和国港口法》 | ✅ |
| 《中华人民共和国国际海运条例》 | ✅ |
| 《中华人民共和国内河交通安全管理条例》 | ✅ |
| 《中华人民共和国防治船舶污染海洋环境管理条例》 | ✅ |

### 国际公约 (5部)
| 法律名称 | 状态 |
|---------|------|
| IMO SOLAS Convention (详细条款) | ✅ |
| IMO MARPOL Convention (详细条款) | ✅ |
| IMO STCW Convention (详细条款) | ✅ |
| ISM Code (国际安全管理规则) | ✅ |
| MLC 2006 (海事劳工公约) | ✅ |

## 快速开始

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载数据 & 构建知识库
PYTHONPATH=. python scripts/build_knowledge_base.py

# 4. 启动交互式问答
PYTHONPATH=. python scripts/run_agent.py
```

## 配置 LLM

### 方式一: OpenAI (推荐)
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-your-key
PYTHONPATH=. python scripts/run_agent.py
```

### 方式二: Ollama (本地部署)
```bash
ollama pull qwen2:7b
export LLM_PROVIDER=ollama
PYTHONPATH=. python scripts/run_agent.py
```

### 无 LLM 模式
不配置 API Key 时自动进入纯检索模式，显示相关法条原文。

## 检索架构

### 混合检索 (Hybrid Search)

三路 RRF (Reciprocal Rank Fusion) 融合，按查询类型动态加权：

| 查询类型 | BM25 | N-gram | Dense | 场景 |
|---------|------|--------|-------|------|
| LATIN (SOLAS/MARPOL等) | 0.15 | 0.10 | 0.75 | 英文缩写 + 中文上下文 |
| BROAD (我国加入/国际海事) | 0.10 | 0.05 | 0.85 | 全量公约查询 |
| PLAIN Chinese | 0.25 | 0.15 | 0.60 | 纯中文法律查询 |

- **BM25**: 基于词频的关键词检索，对中文法律术语精确匹配
- **N-gram (TF-IDF)**: 基于字符 n-gram 的相似度，捕获字面重叠
- **Dense (text-embedding-3-small)**: 1536 维稠密向量语义检索，跨语言匹配
- **Fallback**: Convention/Latin 查询自动 dense 降级补全缺失公约
- **Source Boost**: 查询中显式提及的法源名，RRF 分数 +0.03

### 评估结果 (124 条中文查询, Recall@5)
| 方法 | Hit@1 | Hit@3 | Hit@5 | Hit@10 |
|------|-------|-------|-------|--------|
| **Hybrid** | **91.1%** | **96.0%** | **98.4%** | **100%** |
| Dense only | 77.4% | 90.3% | 96.0% | 96.8% |
| BM25 only | 79.8% | 89.5% | 92.7% | 94.4% |
| N-gram only | 75.8% | 82.3% | 87.9% | 92.7% |

### Reranker
- Cross-encoder: `BAAI/bge-reranker-v2-m3` (精排 top-3)
- Fallback: 基于 jieba 分词 + n-gram 的本地中文重排序
- 支持 OpenAI API embedding 兜底重排

## 项目结构

```
maritime-law-rag/
├── data/
│   ├── raw/           # 原始法律文本
│   ├── processed/     # ChromaDB + BM25 索引
│   └── eval/          # 评估数据集
├── src/
│   ├── data_collection/   # 数据采集
│   ├── document_processing/  # 分块 & 向量化
│   ├── vector_store/      # BM25 + N-gram + Dense 混合存储
│   ├── rag/               # 检索引擎 & 重排序 & 生成
│   └── cli/               # 交互式命令行
├── scripts/
│   ├── build_knowledge_base.py # 构建知识库
│   ├── run_agent.py            # 启动问答
│   ├── eval_retrieval.py       # Recall/MRR 评估
│   └── eval_hitk.py            # Hit@K 评估
└── tests/               # 测试
```
