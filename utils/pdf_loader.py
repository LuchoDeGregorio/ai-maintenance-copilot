from langchain_community.document_loaders import PyPDFLoader
import os


def load_pdfs_from_folder(folder_path):
    
    documents = []

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            file_path = os.path.join(folder_path, file)

            loader = PyPDFLoader(file_path)

            docs = loader.load()

            documents.extend(docs)

    return documents