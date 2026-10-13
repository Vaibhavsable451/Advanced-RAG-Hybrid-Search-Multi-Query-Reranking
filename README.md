# 02 - Advanced RAG: Hybrid Search + Multi-Query + Reranking

Upgrades basic RAG with the three techniques that give the biggest quality jump:

1. **Hybrid retrieval**: BM25 (keywords) + dense vectors, merged with Reciprocal Rank Fusion.
2. **Multi-query expansion**: LLM rewrites the question into several variants to improve recall.
3. **Cross-encoder reranking**: re-scores the top candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`.

```bash
pip install -r requirements.txt && cp .env.example .env
python -m src.pipeline "What is the max meal reimbursement when travelling?"
```
