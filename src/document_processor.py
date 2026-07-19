from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path:str) -> list[dict]:
    reader = PdfReader(file_path)
    result = []
    for i, p in enumerate(reader.pages):
        text = p.extract_text()
        #if there isnt text on first page
        if not text or not text.strip():
            continue
        info = {
            "text": text,
            "page_number": i + 1,
            "source": file_path
        }
        result.append(info)
    
    return result


def split_into_chunks(pages: list[dict], chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    result = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for chunk in chunks:
            result.append({
                "text": chunk,
                "page_number": page["page_number"],
                "source": page["source"]
            })
    return result