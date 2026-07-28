from src.retriever import get_relevant_chunks
from src.vector_store import create_vector_store
from src.document_processor import load_pdf, split_into_chunks

pages = load_pdf("data/Comparisons.pdf")
chunks = split_into_chunks(pages)
vector_store = create_vector_store(chunks)

results = get_relevant_chunks("What is the abstract about?", vector_store)
for r in results:
    print(f"Strona {r.metadata['page_number']}: {r.page_content[:150]}")
    print("---")