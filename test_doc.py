from src.document_processor import load_pdf, split_into_chunks

pages = load_pdf("Comparisons.pdf")
print(f"Wczytano stron: {len(pages)}")
print(f"Pierwsza strona: {pages[0]['text'][:200]}")

chunks = split_into_chunks(pages, chunk_size=500, chunk_overlap=50)

print(f"Utworzono chunków: {len(chunks)}")