from src.document_processor import load_pdf, split_into_chunks

pages = load_pdf("data/Comparisons.pdf")
print(f"Wczytano stron: {len(pages)}")

chunks = split_into_chunks(pages)
print(f"Liczba chunków: {len(chunks)}")
print(f"\nPierwszy chunk:")
print(f"Tekst: {chunks[0]['text'][:200]}")
print(f"Strona: {chunks[0]['page_number']}")
print(f"Źródło: {chunks[0]['source']}")