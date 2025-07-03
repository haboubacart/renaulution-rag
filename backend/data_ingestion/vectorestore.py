from tqdm import tqdm
from langchain.vectorstores import FAISS
from langchain.schema import Document
from logger import get_logger
logger = get_logger()


def create_vectorstore(embedding_model, index_path):
    try:
        dim = len(embedding_model.embed_query("test"))
        logger.info(f"Création d'un index FAISS vide (dim = {dim})")

        dummy_doc = Document(page_content="placeholder", metadata={})
        vectorstore = FAISS.from_documents(
            documents=[dummy_doc],
            embedding=embedding_model,
            normalize_L2=True
        )
        vectorstore.index.reset()  
        vectorstore.docstore._dict.clear()  
        vectorstore.index_to_docstore_id.clear() 


        logger.info(f"Sauvegarde de l'index vide dans: {index_path}")
        vectorstore.save_local(index_path)

        logger.info("Index vide (cosinus) créé avec succès.")
        return vectorstore
    except Exception as e:
        logger.error(f"Erreur lors de la création de l’index vide: {e}", exc_info=True)
        return None


def index_documents(chunks, embedding_model, index_storing_path, batch_size=64):
    vectorstore = FAISS.load_local(index_storing_path, embedding_model, allow_dangerous_deserialization=True)
    try:
        logger.info(f"Début de l’indexation de {len(chunks)} chunks en batches de {batch_size}...")

        logger.info("Calcul des embeddings et indexation par batch avec barre de progression...")
        for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing", unit="batch"):
            batch = chunks[i:i + batch_size]

            logger.info(f"Indexation du batch {i//batch_size + 1}...")
            vectorstore.add_documents(batch)

        logger.info(f"Sauvegarde de l’index local dans : {index_storing_path}")
        vectorstore.save_local(index_storing_path)
        logger.info("Indexation terminée avec succès.")

    except Exception as e:
        logger.error(f"Erreur lors de l’indexation : {e}", exc_info=True)
