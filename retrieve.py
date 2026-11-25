from typing import List, Dict, Any, Optional
from query_pipeline import default_pipeline, RetrievedChunk


def retrieve_documents(
    question: str,
    tenant_id: str = "tenant_001",
    n_results: int = 3,
    token_budget: Optional[int] = None,
) -> List[str]:
    """Retrieve relevant document texts within an optional token budget."""
    chunks = default_pipeline.retrieve_top_k(question, top_k=n_results, tenant_id=tenant_id)
    if token_budget is not None:
        chunks, _, _ = default_pipeline.apply_token_budget(chunks, token_budget=token_budget)
    return [c.text for c in chunks]


def retrieve_with_metadata(
    question: str,
    tenant_id: str = "tenant_001",
    n_results: int = 3,
    token_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Retrieve chunks with metadata, similarity scores, and token usage."""
    raw_chunks = default_pipeline.retrieve_top_k(question, top_k=n_results, tenant_id=tenant_id)
    budget = token_budget if token_budget is not None else default_pipeline.default_budget
    selected_chunks, tokens_used, dropped = default_pipeline.apply_token_budget(
        raw_chunks, token_budget=budget
    )

    return {
        "question": question,
        "chunks_retrieved": len(raw_chunks),
        "chunks_used": len(selected_chunks),
        "chunks_dropped": dropped,
        "tokens_used": tokens_used,
        "token_budget": budget,
        "documents": [
            {
                "id": c.id,
                "text": c.text,
                "source": c.source,
                "page": c.page,
                "similarity": round(c.similarity, 4),
                "distance": round(c.distance, 4),
                "tokens": c.token_count,
            }
            for c in selected_chunks
        ],
    }


if __name__ == "__main__":
    question = "How many casual leaves are available?"
    data = retrieve_with_metadata(question, n_results=3, token_budget=500)

    print(f"Query: {data['question']}")
    print(f"Chunks: {data['chunks_used']} used / {data['chunks_retrieved']} retrieved")
    print(f"Tokens: {data['tokens_used']} / {data['token_budget']} budget\n")

    for i, doc in enumerate(data["documents"], 1):
        print(f"--- Document {i} [{doc['source']} - Page {doc['page']} | Similarity: {doc['similarity']:.1%}] ---")
        print(doc["text"], "\n")
