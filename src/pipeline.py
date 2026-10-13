import sys
from collections import defaultdict
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import get_llm, get_embeddings

def load_chunks():
    docs = []
    for f in Path("data").glob("*.txt"):
        docs += TextLoader(str(f), encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60).split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks

def rrf(result_lists, k=60):
    """Reciprocal Rank Fusion over several ranked lists of Documents."""
    scores, keep = defaultdict(float), {}
    for lst in result_lists:
        for rank, d in enumerate(lst):
            cid = d.metadata["chunk_id"]
            scores[cid] += 1.0 / (k + rank + 1)
            keep[cid] = d
    return [keep[c] for c, _ in sorted(scores.items(), key=lambda x: -x[1])]

class HybridRAG:
    def __init__(self):
        self.chunks = load_chunks()
        self.bm25 = BM25Retriever.from_documents(self.chunks); self.bm25.k = 6
        self.dense = FAISS.from_documents(self.chunks, get_embeddings()).as_retriever(search_kwargs={"k": 6})
        self.llm = get_llm()
        self._reranker = None

    def expand(self, question, n=3):
        prompt = ChatPromptTemplate.from_template(
            "Rewrite the question into {n} different search queries, one per line, no numbering.\nQuestion: {q}")
        out = (prompt | self.llm | StrOutputParser()).invoke({"q": question, "n": n})
        return [question] + [l.strip() for l in out.splitlines() if l.strip()][:n]

    def rerank(self, question, docs, top_n=4):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = self._reranker.predict([(question, d.page_content) for d in docs])
        return [d for d, _ in sorted(zip(docs, scores), key=lambda x: -x[1])][:top_n]

    def retrieve(self, question):
        lists = []
        for q in self.expand(question):
            lists.append(self.bm25.invoke(q)); lists.append(self.dense.invoke(q))
        return self.rerank(question, rrf(lists)[:12])

    def answer(self, question):
        docs = self.retrieve(question)
        ctx = "\n\n".join(f"[chunk {d.metadata['chunk_id']}] {d.page_content}" for d in docs)
        prompt = ChatPromptTemplate.from_template(
            "Answer only from the context; if absent say 'I don't know'. Cite [chunk N].\n\n{ctx}\n\nQ: {q}")
        return (prompt | self.llm | StrOutputParser()).invoke({"ctx": ctx, "q": question}), docs

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How much can I spend on meals?"
    ans, docs = HybridRAG().answer(q)
    print(ans); print("\nUsed chunks:", [d.metadata["chunk_id"] for d in docs])
