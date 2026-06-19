from __future__ import annotations

import argparse
from pathlib import Path

from ai_core.rag_engine import (
    build_rag_context,
    build_rag_index,
    load_rag_documents,
    load_rag_index,
    load_static_knowledge,
    save_retrieval_results,
    search_rag_index,
)


def query_existing_data(
    data_dir: str | Path,
    question: str,
    top_k: int = 5,
    rebuild: bool = False,
) -> list[dict]:
    root = Path(data_dir)
    documents_path = root / "rag_documents.jsonl"
    index_path = root / "rag_index.json"

    if rebuild or not index_path.exists():
        documents = [*load_rag_documents(documents_path), *load_static_knowledge()]
        build_rag_index(documents, index_path)

    index = load_rag_index(index_path)
    results = search_rag_index(index, question, top_k=top_k)
    save_retrieval_results(results, root / "rag_retrieval.json")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query an existing local RAG index")
    parser.add_argument("data_dir", help="Data folder, for example data/raw_data/2330")
    parser.add_argument("question", help="Question or keywords to search")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild rag_index.json before querying")
    parser.add_argument("--show-context", action="store_true", help="Print assembled RAG context")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = query_existing_data(args.data_dir, args.question, top_k=args.top_k, rebuild=args.rebuild)
    print(f"Retrieved: {len(results)}")
    for index, item in enumerate(results, start=1):
        chunk = ""
        if int(item.get("chunk_count", 1) or 1) > 1:
            chunk = f" chunk={int(item.get('chunk_index', 0)) + 1}/{item.get('chunk_count')}"
        print(f"{index}. [{item.get('source', '')}] {item.get('title', '')}{chunk} score={item.get('score', 0)}")
        if item.get("url"):
            print(f"   {item['url']}")
    if args.show_context:
        print("\n=== RAG Context ===")
        print(build_rag_context(results))


if __name__ == "__main__":
    main()
