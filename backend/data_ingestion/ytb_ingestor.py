from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import uuid
from typing import List
from logger import get_logger 

logger = get_logger()

def create_chunks_by_words(documents: List[Document], source_name: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    all_chunks = []
    for doc in documents:
        words = doc.page_content.split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            all_chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "source": source_name,
                    "id": str(uuid.uuid4())
                }
            ))

            start += chunk_size - chunk_overlap  # décalage pour l'overlap

    return all_chunks


def extract_and_split_ytb(url: str, video_name: str, chunk_size=150, chunk_overlap=25) -> List[Document]:
    try:
        logger.info(f"Extraction des sous-titres pour la vidéo : {video_name} ({url})")
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,
            language=["fr", "en"]
        )
        documents = loader.load()
        logger.info(f"Chargement réussi : {len(documents)} documents extraits de {video_name}")

        all_ytb_chunks = create_chunks_by_words(documents, video_name, chunk_size, chunk_overlap)
        logger.info(f"Chunking terminé : {len(all_ytb_chunks)} chunks générés pour {video_name}")

        return all_ytb_chunks

    except Exception as e:
        logger.error(f"Erreur lors de l'extraction ou du chunking de la vidéo '{video_name}': {e}", exc_info=True)
        return []
