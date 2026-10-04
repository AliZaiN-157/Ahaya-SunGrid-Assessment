import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from sungrid.chat import ChatServices
from sungrid.knowledge import (
    LocalEmbeddings,
    QdrantKnowledgeStore,
    ensure_index,
    load_chunks,
)
from sungrid.model import create_answerer, create_classifier, create_eligibility_checker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


@lru_cache(maxsize=1)
def knowledge_components() -> tuple[LocalEmbeddings, QdrantKnowledgeStore]:
    model_choice = os.getenv("EMBEDDING_MODEL", "bge-small-en-v1.5")
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY", "welcome-key"),
            timeout=5,
        )
        client.get_collections()
    except Exception as exc:
        raise RuntimeError(
            "Cannot connect to Qdrant at QDRANT_URL (default http://localhost:6333). "
            "From the repository root, run `docker compose up -d qdrant`."
        ) from exc

    try:
        embeddings = LocalEmbeddings(model_choice)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load embedding model '{model_choice}'. "
            "Check the model name and internet connection."
        ) from exc

    return embeddings, QdrantKnowledgeStore(client, model_choice)


def ingest_documents() -> int:
    chunks = load_chunks(PROJECT_ROOT / "docs")
    if not chunks:
        raise RuntimeError("No knowledge-base documents were found in the docs folder.")

    embeddings, store = knowledge_components()
    try:
        ensure_index(chunks, embeddings, store)
    except Exception as exc:
        raise RuntimeError(
            "Could not build the local document index. "
            "Check the embedding download and Qdrant."
        ) from exc
    return len(chunks)


@lru_cache(maxsize=1)
def create_chat_services() -> ChatServices:
    classify = create_classifier()
    answer = create_answerer()
    check_eligibility = create_eligibility_checker()
    embeddings, store = knowledge_components()
    return ChatServices(
        classify=classify,
        search=lambda question, category: store.search(
            embeddings.embed_query(question), category
        ),
        answer=answer,
        check_eligibility=check_eligibility,
    )
