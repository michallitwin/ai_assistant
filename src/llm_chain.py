import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.retriever import get_relevant_chunks
from langchain_chroma import Chroma


def get_llm(provider: str = "groq"):
    if provider == "groq":
        return ChatGroq(model="llama3-8b-8192", temperature=0.1)
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)


def format_docs(docs: list) -> str:
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page_number", "N/A")
        source = doc.metadata.get("source", "N/A")
        text = doc.page_content
        formatted.append(f"[Source: {source}, Page: {page}]\n{text}")
    return "\n\n---\n\n".join(formatted)


def generate_answer(query: str, vector_store: Chroma, provider: str = "groq") -> str:
    # retrieve and format context from the vector database
    docs = get_relevant_chunks(query=query, vector_store=vector_store, k=3)

    if not docs:
        return "No relevant context found in the documents to answer this query."

    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert analytical assistant. 
        Your task is to answer the user's query STRICTLY based on the provided PDF context below.
        Always cite the source and page number you are using.
        If the context does not contain the information needed to answer the query, state clearly: 'Information not found in the documents'.""",
            ),
            ("human", "Context:\n{context}\n\nQuery: {query}"),
        ]
    )

    llm = get_llm(provider=provider)

    chain = prompt | llm

    response = chain.invoke({"context": context, "query": query})

    return response.content
