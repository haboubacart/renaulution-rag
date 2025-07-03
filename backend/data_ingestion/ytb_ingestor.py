from utils import create_chunks
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
from typing import List
from logger import get_logger 

logger = get_logger()

def extract_and_split_ytb(url: str, video_name: str, chunk_size=450, chunk_overlap=64) -> List[Document]:
    try:
        logger.info(f"Extraction des sous-titres pour la vidéo : {video_name} ({url})")
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,
            language=["fr", "en"]
        )
        documents = loader.load()
        logger.info(f"Chargement réussi : {len(documents)} documents extraits de {video_name}")

        all_ytb_chunks = create_chunks(documents, video_name, chunk_size, chunk_overlap, False)
        logger.info(f"Chunking terminé : {len(all_ytb_chunks)} chunks générés pour {video_name}")

        return all_ytb_chunks

    except Exception as e:
        logger.error(f"Erreur lors de l'extraction ou du chunking de la vidéo '{video_name}': {e}", exc_info=True)
        return []
