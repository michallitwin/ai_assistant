import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.retriever import get_relevant_chunks



def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1
    )


def format_docs(docs: list) -> str:
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page_number", "N/A")
        source = doc.metadata.get("source", "N/A")
        text =
