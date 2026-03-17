import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.pdf_loader import load_pdfs_from_folder
from services.rag_pipeline import split_documents, create_qa_chain
from services.embeddings import create_vector_store

st.title("AI Maintenance Copilot")

if st.button("Procesar documentos"):

    docs = load_pdfs_from_folder("data/manuals")
    chunks = split_documents(docs)
    vectorstore = create_vector_store(chunks)

    st.session_state.qa_chain = create_qa_chain(vectorstore)

    st.success("Documentos procesados correctamente")

if "qa_chain" in st.session_state:

    question = st.text_input("Hacé tu pregunta:")

    if question:

        response = st.session_state.qa_chain(question)

        st.write(response)