from utils.pdf_loader import load_pdfs_from_folder
from services.rag_pipeline import split_documents

docs = load_pdfs_from_folder("data/manuals")
chunks = split_documents(docs)

print(f"Cantidad de documentos: {len(docs)}")
print(f"Cantidad de chunks: {len(chunks)}")

print(chunks[0].page_content)