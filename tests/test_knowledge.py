from pathlib import Path

from sungrid.knowledge import (
    IndexManifest,
    QdrantKnowledgeStore,
    corpus_hash,
    ensure_index,
    load_chunks,
)


def test_loaded_chunks_include_index_metadata():
    chunks = load_chunks(Path("docs"))
    memberships = next(
        chunk
        for chunk in chunks
        if chunk.document_id == "01_program_policies"
        and chunk.section_heading == "Membership Tiers"
    )
    grievance = next(
        chunk
        for chunk in chunks
        if chunk.document_id == "01_program_policies"
        and chunk.section_heading == "Grievance Procedure"
    )
    annual_meeting = next(
        chunk
        for chunk in chunks
        if chunk.document_id == "01_program_policies"
        and chunk.section_heading == "Annual Meeting Attendance"
    )

    assert memberships.chunk_index == 0
    assert memberships.source_path == "01_program_policies.md"
    assert memberships.token_count > 0
    assert grievance.cross_references == ["Incentive & Rebate Programs document"]
    assert annual_meeting.cross_references == [
        "Cooperative Governance & Voting Rights document"
    ]


def test_qdrant_chunk_payload_persists_index_metadata():
    class FakeClient:
        points = []

        def collection_exists(self, collection_name):
            return False

        def create_collection(self, **kwargs):
            pass

        def create_payload_index(self, **kwargs):
            pass

        def upsert(self, collection_name, points, wait):
            self.points = points

    client = FakeClient()
    chunks = load_chunks(Path("docs"))
    chunk = next(
        item
        for item in chunks
        if item.document_id == "01_program_policies"
        and item.section_heading == "Grievance Procedure"
    )
    store = QdrantKnowledgeStore(client, "bge-small-en-v1.5")

    manifest = IndexManifest(
        corpus_hash=corpus_hash(chunks),
        model_id="bge-small-en-v1.5",
        dimensions=384,
        chunk_count=len(chunks),
    )
    store.rebuild(chunks, [[0.0] * 384 for _ in chunks], manifest)

    payload = next(point.payload for point in client.points if point.id == chunk.point_id)
    assert payload["cross_references"] == ["Incentive & Rebate Programs document"]
    assert payload["chunk_index"] == chunk.chunk_index
    assert payload["token_count"] == chunk.token_count
    assert payload["source_path"] == "01_program_policies.md"


def test_unchanged_index_is_reused_without_embedding_again():
    class FakeEmbeddings:
        model_id = "test-model"
        dimensions = 2
        embedded_batches = []

        def embed_documents(self, texts):
            self.embedded_batches.append(texts)
            return [[0.0, 1.0] for _ in texts]

    class FakeStore:
        manifest = None
        rebuild_count = 0

        def is_current(self, manifest):
            return self.manifest == manifest

        def rebuild(self, chunks, vectors, manifest):
            self.manifest = manifest
            self.rebuild_count += 1

    chunks = load_chunks(Path("docs"))
    embeddings = FakeEmbeddings()
    store = FakeStore()

    ensure_index(chunks, embeddings, store)
    ensure_index(chunks, embeddings, store)

    assert len(embeddings.embedded_batches) == 1
    assert store.rebuild_count == 1


def test_index_hash_changes_when_persisted_metadata_changes():
    chunks = load_chunks(Path("docs"))
    changed = chunks[0].model_copy(
        update={"source_path": "updated/" + chunks[0].source_path}
    )

    assert corpus_hash(chunks) != corpus_hash([changed, *chunks[1:]])
