import argparse
import sys

import rag


def ingest(rebuild=False):
    records = rag.load_records()
    chunks = rag.build_chunks(records)

    if rebuild:
        import chromadb

        client = chromadb.PersistentClient(path=str(rag.CHROMA_DIR))
        try:
            client.delete_collection(rag.COLLECTION_NAME)
        except Exception:
            pass
        rag._collection = None

    collection = rag.get_collection()
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["document"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )

    return records, chunks, collection.count()


def main():
    parser = argparse.ArgumentParser(description="Build the habit-research vector store.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing collection before ingesting.",
    )
    args = parser.parse_args()

    records, chunks, total = ingest(rebuild=args.rebuild)

    print(f"knowledge file : {rag.KNOWLEDGE_FILE}")
    print(f"vector store   : {rag.CHROMA_DIR}")
    print(f"collection     : {rag.COLLECTION_NAME}")
    print(f"sources        : {len(records)}")
    print(f"chunks written : {len(chunks)}")
    print(f"chunks in store: {total}")

    if total != len(chunks):
        print(
            f"\n[warning] store holds {total} chunks but this run wrote {len(chunks)}. "
            "Run with --rebuild to drop stale chunks.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
