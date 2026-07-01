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
- **BM25 (稀疏检索)**: 基于词频的统计检索，对中文法律术语效果好
- **TF-IDF + Cosine Similarity**: 基于字符 n-gram 的向量检索
- **融合策略**: 加权融合 BM25 + TF-IDF 分数 (默认 alpha=0.5)

### 交互式命令
- `quit/exit` - 退出
- `context` - 切换显示检索详情
- `hybrid` - 切换混合/纯BM25检索

## 项目结构

```
maritime-law-rag/
├── data/
│   ├── raw/           # 原始法律文本
│   └── processed/     # 向量索引文件
├── src/
│   ├── data_collection/   # 数据采集
│   ├── document_processing/  # 分块 & 向量化
│   ├── vector_store/      # BM25 + TF-IDF 混合存储
│   ├── rag/               # 检索引擎 & 生成引擎
│   └── cli/               # 交互式命令行
├── scripts/
│   ├── build_knowledge_base.py # 构建知识库
│   └── run_agent.py            # 启动问答
└── .env                 # 配置文件
```

## 后续计划
- [ ] 添加多语言 Dense Embedding (paraphrase-multilingual-MiniLM-L12-v2) 提升语义检索
- [ ] 添加 Cross-encoder Reranker (bge-reranker) 精排
- [ ] 集成 ChromaDB 持久化向量存储
- [ ] 支持 PDF/扫描件 OCR
