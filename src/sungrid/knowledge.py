import hashlib
import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

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


class IndexManifest(BaseModel):
    corpus_hash: str
    model_id: str
    dimensions: int
    chunk_count: int


def load_chunks(documents_dir: Path) -> list[KnowledgeChunk]:
    chunks = []
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
        for section in sections[1:]:
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
            chunks.append(
                KnowledgeChunk(
                    point_id=str(uuid5(NAMESPACE_URL, chunk_key)),
                    document_id=path.stem,
                    document_title=title,
                    section_heading=heading,
                    heading_path=heading,
                    body=body,
                    embedding_text=f"{title}\n{heading}\n{body}",
                    score=0.0,
                    primary_category=categories[0],
                    categories=list(categories),
                )
            )
    return chunks


def corpus_hash(chunks: list[KnowledgeChunk]) -> str:
    contents = "\n".join(
        f"{chunk.document_id}:{chunk.document_title}:{chunk.section_heading}:{chunk.body}:{','.join(chunk.categories)}"
        for chunk in chunks
    )
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()


class LocalEmbeddings:
    MODELS = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
        "nomic-embed-text-v1.5": "nomic-ai/nomic-embed-text-v1.5",
    }

    def __init__(self, model_choice: str = "bge-small-en-v1.5") -> None:
        if model_choice not in self.MODELS:
            raise ValueError(f"Unknown embedding model: {model_choice}")

        from sentence_transformers import SentenceTransformer

        model_id = self.MODELS[model_choice]
        self.model = SentenceTransformer(
            model_id,
            trust_remote_code=model_choice == "nomic-embed-text-v1.5",
        )
        self.model_id = model_id
        self.model_choice = model_choice
        dimensions = self.model.get_sentence_embedding_dimension()
        if dimensions is None:
            raise RuntimeError(
                f"Embedding model '{model_id}' did not report its vector size."
            )
        self.dimensions = int(dimensions)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.model_choice == "nomic-embed-text-v1.5":
            texts = [f"search_document: {text}" for text in texts]
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        if self.model_choice == "bge-small-en-v1.5":
            text = f"Represent this sentence for searching relevant passages: {text}"
        elif self.model_choice == "nomic-embed-text-v1.5":
            text = f"search_query: {text}"
        return self.model.encode(text, normalize_embeddings=True).tolist()


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
    embeddings: LocalEmbeddings,
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
