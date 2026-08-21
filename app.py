import os
import streamlit as st
from src.document_processor import load_pdf, split_into_chunks
from src.vector_store import create_vector_store
from src.llm_chain import generate_answer
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF RAG System", layout="wide")
st.title("PDF RAG Analyzer")

with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process & Index"):
            with st.spinner("Loading, chunking, and generating embeddings..."):
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                pages = load_pdf(temp_path)
                chunks = split_into_chunks(pages)
                vector_store = create_vector_store(chunks)

                st.session_state["vector_store"] = vector_store

                if os.path.exists(temp_path):
                    os.remove(temp_path)

                st.success(f"Successfully indexed {len(chunks)} chunks from {len(pages)} pages!")


    st.divider()
    st.header("2. Choose LLM Model")
    selected_model_label = st.selectbox(
        "Select Model Provider:",
        options=["Groq (Llama 3.1)", "Google (Gemini 1.5 Flash)"],
        index=0
    )
    
st.header("2. Ask a Question")

if "vector_store" in st.session_state:
    query = st.text_input("Enter your question about the document:")
    
    if st.button("Generate Answer"):
        if query.strip():
            with st.spinner("Searching database and generating answer..."):
                answer = generate_answer(
                    query=query, 
                    vector_store=st.session_state["vector_store"]
                )
                st.markdown("### Answer:")
                st.write(answer)
        else:
            st.warning("Please enter a question.")
else:
    st.info("Upload and process a PDF file in the sidebar to start asking questions.")