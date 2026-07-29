import os
import streamlit as st
from src.document_processor import load_pdf, split_into_chunks
from src.vector_store import create_vector_store
from src.generator import generate_answer


st.set_page_config(page_title="PDF RAG System", layout="wide")
st.title("PDF RAG Analyzer")

with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    