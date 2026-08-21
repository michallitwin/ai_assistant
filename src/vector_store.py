import os
import shutil
from langchain_chroma import Chroma
from src.embeddings import get_embedding_model

CHROMA_PATH = "chroma_db"


def create_vector_store(chunks: list[dict]) -> Chroma:
    if os.path.exists(CHROMA_PATH):
        for filename in os.listdir(CHROMA_PATH):
            file_path = os.path.join(CHROMA_PATH, filename)

            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except OSError as e:
                print(f"Could not remove {file_path}: {e}")
    else:
        os.makedirs(CHROMA_PATH)

    texts = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "page_number": chunk["page_number"],
            "source": chunk["source"],
        }
        for chunk in chunks
    ]

    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=get_embedding_model(),
        metadatas=metadatas,
        persist_directory=CHROMA_PATH,
    )

    return vector_store
