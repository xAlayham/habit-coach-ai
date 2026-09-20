import pytest

import agent
import rag


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def count(self):
        return len(self.rows)

    def query(self, query_texts, n_results):
        self.queries.append({"query_texts": query_texts, "n_results": n_results})
        rows = self.rows[:n_results]
        return {
            "documents": [[row["document"] for row in rows]],
            "metadatas": [[row["metadata"] for row in rows]],
            "distances": [[row["distance"] for row in rows]],
        }


def row(doc_id, distance, topic="topic"):
    return {
        "document": f"excerpt from {doc_id}",
        "distance": distance,
        "metadata": {
            "doc_id": doc_id,
            "topic": topic,
            "source": f"{doc_id} source",
            "url": f"https://example.test/{doc_id}",
            "kind": "peer-reviewed",
            "chunk_index": 0,
        },
    }


@pytest.fixture
def fake_store(monkeypatch):
    def install(rows):
        collection = FakeCollection(rows)
        monkeypatch.setattr(rag, "_collection", collection)
        return collection

    return install


def test_short_passage_stays_one_chunk():
    text = " ".join(["word"] * 40)

    assert rag.chunk_passage(text, chunk_words=70, overlap_words=20) == [text]


def test_long_passage_splits_with_overlap():
    words = [f"w{i}" for i in range(200)]
    chunks = rag.chunk_passage(" ".join(words), chunk_words=70, overlap_words=20)

    assert len(chunks) > 1
    first = chunks[0].split()
    second = chunks[1].split()
    assert len(first) == 70
    assert first[-20:] == second[:20]


def test_chunking_loses_no_words():
    words = [f"w{i}" for i in range(200)]
    chunks = rag.chunk_passage(" ".join(words), chunk_words=70, overlap_words=20)

    seen = []
    for chunk in chunks:
        for word in chunk.split():
            if word not in seen:
                seen.append(word)

    assert seen == words


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        rag.chunk_passage("a b c", chunk_words=10, overlap_words=10)


def test_corpus_is_well_formed():
    records = rag.load_records()

    assert 10 <= len(records) <= 20
    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids))
    for record in records:
        assert set(record) >= {"id", "topic", "source", "url", "kind", "text"}
        assert record["text"].strip()
        assert record["source"].strip()


def test_every_chunk_keeps_its_citation():
    records = rag.load_records()
    chunks = rag.build_chunks(records)
    sources = {record["id"]: record["source"] for record in records}

    assert len(chunks) >= len(records)
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        metadata = chunk["metadata"]
        assert metadata["source"] == sources[metadata["doc_id"]]
        assert chunk["id"] == f"{metadata['doc_id']}::{metadata['chunk_index']}"


def test_search_returns_one_hit_per_source(fake_store):
    fake_store([
        row("lally", 0.30),
        row("lally", 0.31),
        row("lally", 0.33),
        row("fogg", 0.40),
        row("clear", 0.45),
    ])

    hits = rag.search_research("broke my streak", top_k=3)

    assert [hit["source"] for hit in hits] == [
        "lally source",
        "fogg source",
        "clear source",
    ]


def test_search_overfetches_to_survive_deduplication(fake_store):
    collection = fake_store([row(f"doc{i}", 0.1 * i) for i in range(20)])

    rag.search_research("anything", top_k=3)

    assert collection.queries[0]["n_results"] == 3 * rag.OVERFETCH


def test_search_reports_similarity_not_distance(fake_store):
    fake_store([row("lally", 0.25)])

    hits = rag.search_research("anything", top_k=1)

    assert hits[0]["similarity"] == 0.75


def test_top_k_is_clamped(fake_store):
    fake_store([row(f"doc{i}", 0.1) for i in range(20)])

    assert len(rag.search_research("anything", top_k=99)) == 5
    assert len(rag.search_research("anything", top_k=0)) == 1


def test_empty_store_raises(fake_store):
    fake_store([])

    with pytest.raises(rag.EmptyKnowledgeBase, match="ingest.py"):
        rag.search_research("anything")


def test_research_tool_is_registered_without_the_token():
    tool_names = [tool["name"] for tool in agent.TOOLS]
    assert "search_habit_research" in tool_names

    functions = agent.build_tool_functions("secret-jwt")
    assert functions["search_habit_research"] is agent.search_habit_research
    assert "secret-jwt" not in str(agent.search_habit_research_tool)


@pytest.mark.anyio
async def test_agent_can_call_the_research_tool(habit_api, use_model, monkeypatch):
    from fakes import calls_tools, says

    captured = {}

    async def fake_search(query, top_k=3):
        captured["query"] = query
        captured["top_k"] = top_k
        return [{"excerpt": "Missing one day did not matter.", "source": "Lally (2010)"}]

    monkeypatch.setattr(agent, "search_habit_research", fake_search)
    model = use_model(
        calls_tools(("search_habit_research", {"query": "broke my streak"})),
        says("One missed day is fine (Lally et al., 2010)."),
    )

    answer = await agent.run_agent("I broke my streak", "jwt-abc")

    assert answer == "One missed day is fine (Lally et al., 2010)."
    assert captured["query"] == "broke my streak"
    assert habit_api.requests == []
    result = model.sent_messages(1)[-1]["content"][0]
    assert result["is_error"] is False
    assert "Lally" in result["content"]


@pytest.mark.anyio
async def test_missing_knowledge_base_surfaces_as_a_tool_error(habit_api, use_model, monkeypatch):
    from fakes import calls_tools, says

    async def fake_search(query, top_k=3):
        raise rag.EmptyKnowledgeBase("The research knowledge base is empty.")

    monkeypatch.setattr(agent, "search_habit_research", fake_search)
    model = use_model(
        calls_tools(("search_habit_research", {"query": "anything"})),
        says("I cannot reach my research library right now."),
    )

    await agent.run_agent("how long do habits take?", "jwt-abc")

    result = model.sent_messages(1)[-1]["content"][0]
    assert result["is_error"] is True
    assert "knowledge base is empty" in result["content"]
