from __future__ import annotations

from metadata_indexer import MetadataIndexer
from monitor_drive import DriveMonitor
from semantic_search import SemanticSearchEngine


def run_test_flow(sample_uploader: str = "Amber") -> None:
    # 1) Pull latest files and index changes.
    monitor = DriveMonitor()
    changes = monitor.run_once()
    print(f"Sync complete: {changes}")

    # 2) Build search engine over indexed metadata.
    indexer = MetadataIndexer()
    indexer.seed_dummy_data_if_empty()
    search = SemanticSearchEngine(indexer)

    # 3) Run sample natural language query.
    query = f"Show files uploaded by {sample_uploader}"
    results = search.search(query, top_k=10)
    print(f"\nQuery: {query}")
    print(f"Matches: {len(results)}")
    for r in results[:5]:
        print(f"- {r.get('file_name')} | uploader={r.get('uploader_name')} | modified={r.get('modified_time')}")


if __name__ == "__main__":
    run_test_flow()
