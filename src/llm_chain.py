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
        text = doc.page_content
        formatted.append(f"[Source: {source}, Page: {page}]\n{text}")
    return "\n\n---\n\n".join(formatted)


def generate_answer(query: str, vector_store: Chroma) -> str:
    #retrieve and format context from the vector database
    docs = get_relevant_chunks(query=query, vector_store=vector_store, k=3)

    if not docs:
        return "No relevant context found in the documents to answer this query."

    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert analytical assistant. 
        Your task is to answer the user's query STRICTLY based on the provided PDF context below.
        Always cite the source and page number you are using.
        If the context does not contain the information needed to answer the query, state clearly: 'Information not found in the documents'."""),
        ("human", "Context:\n{context}\n\nQuery: {query}")
    ])


    llm = get_llm()

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "query": query
    })

    return response.content