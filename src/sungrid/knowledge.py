import hashlib
import os
import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import httpx
from pydantic import BaseModel

from sungrid.chat import RetrievedChunk
from sungrid.taxonomy import DOCUMENT_CATEGORIES


class KnowledgeChunk(RetrievedChunk):
    point_id: str
    document_id: str
    heading_path: str
    embedding_text: str
    primary_category: str
    categories: list[str]
    cross_references: list[str]
    chunk_index: int
    token_count: int
    source_path: str


class IndexManifest(BaseModel):
    corpus_hash: str
    model_id: str
    dimensions: int
    chunk_count: int


def load_chunks(documents_dir: Path) -> list[KnowledgeChunk]:
    chunks = []
    cross_reference_patterns = (
        re.compile(r"\(see\s+([^)]+)\)", re.IGNORECASE),
        re.compile(
            r"\b([A-Z][A-Za-z0-9'’.-]*(?:\s+(?:[A-Z][A-Za-z0-9'’.-]*|&))*\s+(?:document|materials))\b"
        ),
    )
    token_pattern = re.compile(r"\w+|[^\w\s]")
    for path in sorted(documents_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = next(
            (
                line.lstrip("# ").strip()
                for line in text.splitlines()
                if line.startswith("# ")
            ),
            path.stem,
        )
        sections = re.split(r"(?m)^##\s+", text)
        for chunk_index, section in enumerate(sections[1:]):
            heading, _, body = section.partition("\n")
            body = body.strip()
            if not heading.strip() or not body:
                continue
            heading = heading.strip()
            categories = DOCUMENT_CATEGORIES.get(path.stem)
            if not categories:
                raise ValueError(
                    f"No category mapping for knowledge document: {path.name}"
                )
            chunk_key = f"{path.name}:{heading}:{body}"
            embedding_text = f"{title}\n{heading}\n{body}"
            cross_reference_matches = sorted(
                (
                    match.start(),
                    match.end(),
                    match.group(1).strip().rstrip(".,;"),
                )
                for pattern in cross_reference_patterns
                for match in pattern.finditer(body)
            )
            cross_references = []
            covered_until = -1
            for start, end, reference in cross_reference_matches:
                if start >= covered_until and reference not in cross_references:
                    cross_references.append(reference)
                    covered_until = end
            chunks.append(
                KnowledgeChunk(
                    point_id=str(uuid5(NAMESPACE_URL, chunk_key)),
                    document_id=path.stem,
                    document_title=title,
                    section_heading=heading,
                    heading_path=heading,
                    body=body,
                    embedding_text=embedding_text,
                    score=0.0,
                    primary_category=categories[0],
                    categories=list(categories),
                    cross_references=cross_references,
                    chunk_index=chunk_index,
                    token_count=len(token_pattern.findall(embedding_text)),
                    source_path=path.relative_to(documents_dir).as_posix(),
                )
            )
    return chunks


def corpus_hash(chunks: list[KnowledgeChunk]) -> str:
    contents = "\n".join(
        f"{chunk.document_id}:{chunk.document_title}:{chunk.section_heading}:{chunk.body}:"
        f"{','.join(chunk.categories)}:{','.join(chunk.cross_references)}:"
        f"{chunk.chunk_index}:{chunk.token_count}:{chunk.source_path}"
        for chunk in chunks
    )
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()


class OpenRouterEmbeddings:
    DEFAULT_MODEL = "openai/text-embedding-3-small"
    DIMENSIONS = 1536
    ENDPOINT = "https://openrouter.ai/api/v1/embeddings"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        api_key = (
            os.getenv("OPENROUTER_API_KEY", "") if api_key is None else api_key
        )
        if not api_key.strip():
            raise ValueError(
                "Set OPENROUTER_API_KEY in .env before starting the API service."
            )

        self.model_id = model_id
        self.dimensions = self.DIMENSIONS
        self.api_key = api_key.strip()
        self.client = client or httpx.Client(timeout=30)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        from sungrid.model import retry_model_call

        def request():
            response = self.client.post(
                self.ENDPOINT,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model_id,
                    "input": texts,
                    "dimensions": self.dimensions,
                },
            )
            response.raise_for_status()
            return response

        response = retry_model_call(request)
        items = response.json()["data"]
        if len(items) != len(texts):
            raise RuntimeError(
                "OpenRouter returned an unexpected number of embeddings."
            )
        return [
            item["embedding"]
            for item in sorted(items, key=lambda item: item["index"])
        ]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class QdrantKnowledgeStore:
    def __init__(self, client, model_choice: str) -> None:
        self.client = client
        self.collection_name = "sungrid_" + re.sub(
            r"[^a-z0-9]+", "_", model_choice.lower()
        ).strip("_")

    def is_current(self, manifest: IndexManifest) -> bool:
        if not self.client.collection_exists(self.collection_name):
            return False
        from qdrant_client import models

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="record_type", match=models.MatchValue(value="manifest")
                    )
                ]
            ),
            limit=1,
            with_payload=True,
        )
        return bool(
            points
            and points[0].payload
            == {"record_type": "manifest", **manifest.model_dump()}
        )

    def rebuild(
        self,
        chunks: list[KnowledgeChunk],
        vectors: list[list[float]],
        manifest: IndexManifest,
    ) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=manifest.dimensions, distance=models.Distance.COSINE
            ),
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="categories",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        points = [
            models.PointStruct(
                id=chunk.point_id,
                vector=vector,
                payload={
                    "record_type": "chunk",
                    "document_id": chunk.document_id,
                    "document_title": chunk.document_title,
                    "section_heading": chunk.section_heading,
                    "heading_path": chunk.heading_path,
                    "body": chunk.body,
                    "primary_category": chunk.primary_category,
                    "categories": chunk.categories,
                    "cross_references": chunk.cross_references,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "source_path": chunk.source_path,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        manifest_id = str(uuid5(NAMESPACE_URL, self.collection_name + ":manifest"))
        points.append(
            models.PointStruct(
                id=manifest_id,
                vector=[0.0] * manifest.dimensions,
                payload={"record_type": "manifest", **manifest.model_dump()},
            )
        )
        self.client.upsert(
            collection_name=self.collection_name, points=points, wait=True
        )

    def search(
        self, vector: list[float], category: str | None = None, limit: int = 4
    ) -> list[RetrievedChunk]:
        from qdrant_client import models

        filters: list[models.Condition] = [
            models.FieldCondition(
                key="record_type", match=models.MatchValue(value="chunk")
            )
        ]
        if category:
            filters.append(
                models.FieldCondition(
                    key="categories", match=models.MatchValue(value=category)
                )
            )
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=models.Filter(must=filters),
            limit=limit,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                chunk_id=str(hit.id),
                document_title=hit.payload["document_title"],
                section_heading=hit.payload["section_heading"],
                body=hit.payload["body"],
                score=hit.score,
            )
            for hit in hits
        ]


def ensure_index(
    chunks: list[KnowledgeChunk],
    embeddings: OpenRouterEmbeddings,
    store: QdrantKnowledgeStore,
) -> None:
    manifest = IndexManifest(
        corpus_hash=corpus_hash(chunks),
        model_id=embeddings.model_id,
        dimensions=embeddings.dimensions,
        chunk_count=len(chunks),
    )
    if store.is_current(manifest):
        return
    vectors = embeddings.embed_documents([chunk.embedding_text for chunk in chunks])
    store.rebuild(chunks, vectors, manifest)
