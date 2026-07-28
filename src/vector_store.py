from langchain_chroma import Chroma
from src.embeddings import get_embedding_model
import shutil
import os
import chromadb

#directory where chromadb vector database will be stored locally
CHROMA_PATH = "chroma_db"

def create_vector_store(chunks: list[dict]) -> Chroma:

    #delete old data
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        
    #extract plain texts for embedding generation
    texts = [chunk["text"] for chunk in chunks]

    metadatas = [
        {"page_number": chunk["page_number"], "source": chunk["source"]} 
        for chunk in chunks
    ]
    #load our embedding model instance
    embedding_function = get_embedding_model()

    #generate vectors and save on a disk 
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embedding_function,
        metadatas=metadatas,
        persist_directory=CHROMA_PATH
    )

    return vector_store 