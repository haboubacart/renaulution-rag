from utils import create_chunks
from langchain.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from typing import List
from logger import get_logger  
import os  

logger = get_logger()

def extract_and_split_pdf(file_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    filename = os.path.basename(file_path)
    try:
        logger.info(f"Chargement du PDF : {filename}")
        loader = UnstructuredPDFLoader(
            file_path,
            strategy="fast"
        )
        documents = loader.load()
        logger.info(f"PDF chargé avec succès : {filename} ({len(documents)} pages)")

        all_pdf_chunks = create_chunks(documents, filename, chunk_size, chunk_overlap)
        logger.info(f"Chunking terminé : {len(all_pdf_chunks)} chunks générés")
        return all_pdf_chunks

    except Exception as e:
        logger.error(f"Erreur lors du traitement du PDF '{filename}': {e}", exc_info=True)
        return []
