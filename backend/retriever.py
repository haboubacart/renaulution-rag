import os
import pickle
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

# Dédupliquer les documents par contenu
def deduplicate_documents(documents: list[Document]) -> list[Document]:
    seen = set()
    unique_docs = []
    for doc in documents:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)
    return unique_docs

# Charger le modèle d'embedding
def load_embedding_model(model_path: str):
    return HuggingFaceBgeEmbeddings(
        model_name=model_path,
        encode_kwargs={"normalize_embeddings": True},
        query_instruction="query: "
    )

# Charger le vectorstore FAISS
def load_faiss_vectorstore(index_path: str, embedding_model) -> FAISS:
    return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

# Charger les documents picklés
def load_documents(pickle_path: str) -> list[Document]:
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

# Créer le retriever sémantique
def create_semantic_retriever(vectorstore: FAISS, k: int = 4):
    return vectorstore.as_retriever(search_kwargs={"k": k})

# Créer le retriever lexical BM25
def create_lexical_retriever(documents: list[Document], k: int = 1):
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = k
    return bm25

# Créer un retriever hybride
def create_hybrid_retriever(semantic_retriever, lexical_retriever, weights: list = [0.8, 0.2]):
    return EnsembleRetriever(
        retrievers=[semantic_retriever, lexical_retriever],
        weights=weights
    )

# Chargement complet du retriever hybride
def load_hybrid_retriever(
    index_path: str,
    pickle_path: str,
    model_path: str,
    k_semantic: int = 7,
    k_lexical: int = 2,
    weights: list = [0.8, 0.2]
):
    embedding_model = load_embedding_model(model_path)
    vectorstore = load_faiss_vectorstore(index_path, embedding_model)
    semantic_retriever = create_semantic_retriever(vectorstore, k=k_semantic)

    documents = load_documents(pickle_path)
    lexical_retriever = create_lexical_retriever(documents, k=k_lexical)

    return create_hybrid_retriever(semantic_retriever, lexical_retriever, weights)

