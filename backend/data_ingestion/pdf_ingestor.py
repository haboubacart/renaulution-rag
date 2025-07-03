from utils import create_chunks
from langchain.document_loaders import UnstructuredPDFLoader
import os  


def extract_and_split_pdf(file_path: str, chunk_size: int, chunk_overlap: int) -> list:
    # Chargement du PDF avec structure
    filename = os.path.basename(file_path)
    loader = UnstructuredPDFLoader(file_path, 
                                   strategy="auto")
    documents = loader.load()  # Chaque page est un Document avec metadata
    all_pdf_chunks = create_chunks(documents, filename, chunk_size, chunk_overlap)
    
    return all_pdf_chunks
