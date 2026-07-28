from langchain_chroma import Chroma


def get_relevant_chunks(query: str, vector_store: Chroma, k: int = 3) -> list:
    results = vector_store.similarity_search(query, k=k)
    return results