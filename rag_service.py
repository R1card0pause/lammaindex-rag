from __future__ import annotations

import os
from pathlib import Path

import requests
import yaml
from fastapi import FastAPI
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
from pydantic import BaseModel, PrivateAttr


SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"


class SiliconFlowEmbedding(BaseEmbedding):
    model: str
    api_key: str
    batch_size: int = 64
    timeout: int = 120
    _session: requests.Session = PrivateAttr(default_factory=requests.Session)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._session.post(
            f"{SILICONFLOW_BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed_batch([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            embeddings.extend(self._embed_batch(texts[i : i + self.batch_size]))
        return embeddings


def expand_env(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def chat_completion(model: str, api_key: str, question: str, contexts: list[str]) -> str:
    context_text = "\n\n---\n\n".join(contexts)
    response = requests.post(
        f"{SILICONFLOW_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个严谨的中文RAG助手。只根据给定资料回答；资料不足时说明无法从资料中确定。",
                },
                {
                    "role": "user",
                    "content": f"资料：\n{context_text}\n\n问题：{question}",
                },
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


def create_app() -> FastAPI:
    config_path = Path(os.environ.get("RAG_MODEL_CONFIG", "./models.yaml"))
    storage_dir = Path(os.environ.get("RAG_STORAGE_DIR", "./runs/latest/storage"))
    cfg = load_yaml(config_path)
    embed_cfg = cfg["embeddings"][cfg["retrieval"]["file_search_embed_key"]]
    llm_cfg = cfg["llm"]

    embed_key = expand_env(str(embed_cfg["api_key"]))
    llm_key = expand_env(str(llm_cfg["api_key"]))
    if not embed_key or not llm_key:
        raise RuntimeError("Set LAZYLLM_SILICONFLOW_API_KEY before starting the service.")

    Settings.embed_model = SiliconFlowEmbedding(
        model=embed_cfg["model"],
        api_key=embed_key,
        batch_size=64,
    )
    Settings.llm = None

    storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
    index = load_index_from_storage(storage_context)

    app = FastAPI(title="Local LlamaIndex RAG Benchmark Service")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "storage_dir": str(storage_dir)}

    @app.post("/retrieve")
    def retrieve(req: QueryRequest) -> dict:
        retriever = index.as_retriever(similarity_top_k=req.top_k)
        nodes = retriever.retrieve(req.question)
        return {
            "question": req.question,
            "sources": [
                {
                    "rank": i + 1,
                    "score": node.score,
                    "metadata": node.node.metadata,
                    "text": node.node.get_content()[:1200],
                }
                for i, node in enumerate(nodes)
            ],
        }

    @app.post("/query")
    def query(req: QueryRequest) -> dict:
        retriever = index.as_retriever(similarity_top_k=req.top_k)
        nodes = retriever.retrieve(req.question)
        answer = chat_completion(
            model=llm_cfg["model"],
            api_key=llm_key,
            question=req.question,
            contexts=[node.node.get_content()[:2000] for node in nodes],
        )
        return {
            "answer": answer,
            "sources": [
                {
                    "rank": i + 1,
                    "score": node.score,
                    "metadata": node.node.metadata,
                    "text": node.node.get_content()[:800],
                }
                for i, node in enumerate(nodes)
            ],
        }

    return app


app = create_app()
