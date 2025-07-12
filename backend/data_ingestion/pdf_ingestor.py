from utils import create_chunks
from langchain.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from typing import List
from logger import get_logger 
from langchain_docling import DoclingLoader
from typing import List, Tuple 
import uuid
import re
import os  


logger = get_logger()
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

def extract_and_split_pdf(file_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    filename = os.path.basename(file_path)
    try:
        logger.info(f"Chargement du PDF : {filename}")
        loader = UnstructuredPDFLoader(
            file_path,
            strategy="fast"
        )
        documents = loader.load()
        print(documents)
        logger.info(f"PDF chargé avec succès : {filename} ({len(documents)} pages)")

        all_pdf_chunks = create_chunks(documents, filename, chunk_size, chunk_overlap)
        logger.info(f"Chunking terminé : {len(all_pdf_chunks)} chunks générés")
        return all_pdf_chunks

    except Exception as e:
        logger.error(f"Erreur lors du traitement du PDF '{filename}': {e}", exc_info=True)
        return []
    

def clean_markdown_lines(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    cleaned_lines = []

    for line in lines:
        # Supprimer les espaces en début/fin + remplacer plusieurs espaces par un seul
        stripped = re.sub(r'\s+', ' ', line.strip())

        # Ignorer les lignes vides ou sans contenu pertinent
        if not stripped or re.fullmatch(r"[\d\.\-–—•]+", stripped) or stripped in {"I", "II", "III"}:
            continue

        # Supprimer les lignes majuscules très courtes (parasites visuels)
        if stripped.isupper() and len(stripped) < 10:
            continue

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)

def load_pdf_as_markdown(file_path: str, output_md_path: str) -> str:
    """
    Charge un PDF avec Docling, extrait le contenu en Markdown,
    le sauvegarde et retourne le texte markdown.
    """
    logger.info(f"========== Chargement du PDF : {file_path} =================")
    
    loader = DoclingLoader(file_path=file_path, export_type="markdown")
    docs = loader.load()

    markdown_content = docs[0].page_content
    markdown_content = clean_markdown_lines(markdown_content)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    logger.info(f"Markdown sauvegardé dans : {output_md_path}")
    return markdown_content


def split_markdown_into_sections(md_text: str, min_words: int = 30) -> List[Tuple[str, str]]:
    """
    Découpe le markdown en sections (titre + contenu), ignore les sections vides,
    et fusionne les sections trop courtes (hors tableaux) avec la suivante.
    Une section courte contenant une table (lignes commençant par '|') est conservée telle quelle.
    Le titre conservé est celui de la section longue.
    """
    pattern = r"(##+ .+)"
    parts = re.split(pattern, md_text)
    raw_sections = []

    logger.info("Découpage initial du markdown en sections...")

    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip() if (i + 1) < len(parts) else ""

        # On ignore complètement les sections sans contenu
        if not content:
            logger.debug(f"Section ignorée (vide) : {title}")
            continue
        raw_sections.append((title, content))
    logger.info(f"{len(raw_sections)} sections détectées après nettoyage.")

    fused_sections = []
    i = 0
    while i < len(raw_sections):
        current_title, current_content = raw_sections[i]
        current_words = len(current_content.split())
        has_table = any(line.strip().startswith("|") for line in current_content.splitlines())

        # Cas spécial : section courte SANS table → fusion avec la suivante
        if current_words < min_words and not has_table and i + 1 < len(raw_sections):
            next_title, next_content = raw_sections[i + 1]
            logger.debug(f"Fusion de '{current_title}' dans '{next_title}' (section trop courte sans table)")
            fused_content = f"{current_content}\n{next_content}".strip()
            fused_sections.append((next_title, fused_content))
            i += 2
        else:
            fused_sections.append((current_title, current_content))
            i += 1

    logger.info(f"{len(fused_sections)} sections finales prêtes.")
    return fused_sections


def extract_tables(text: str) -> Tuple[List[str], str]:
    table_pattern = r"((?:\|.*\n)+)"
    tables = re.findall(table_pattern, text)
    cleaned_text = re.sub(table_pattern, "", text)
    
    if tables:
        logger.info(f"{len(tables)} table(s) détectée(s).")
    return tables, cleaned_text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def chunk_markdown_with_table(md_text: str, chunk_size:int, overlap: int) -> List[str]:
    final_chunks = []
    sections = split_markdown_into_sections(md_text)

    logger.info("Début du découpage par section avec gestion des tables...")
    for idx, (title, content) in enumerate(sections):
        tables, rest = extract_tables(content)
        if rest:
            text_chunks = chunk_text(rest, chunk_size, overlap)
            for chunk in text_chunks:
                final_chunks.append(f"{title}\n{chunk}")

        for table in tables:
            final_chunks.append(f"{title}\n{table.strip()}")

    logger.info(f"Total des chunks générés : {len(final_chunks)}")
    return final_chunks

def extract_and_split_pdf_v2(file_path: str, output_md_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    filename = os.path.basename(file_path)
    all_chunks = []
    try:
        markdown_content = load_pdf_as_markdown(file_path,  output_md_path)
        all_pdf_chunks = chunk_markdown_with_table(markdown_content, chunk_size, chunk_overlap)
        for chunk in all_pdf_chunks : 
            all_chunks.append(Document(
                    page_content = chunk,
                    metadata={
                        "source": filename.split('.')[0],
                        "id": str(uuid.uuid4())
                    }
                ))
        return all_chunks

    except Exception as e:
        logger.error(f"Erreur lors du traitement du PDF '{filename}': {e}", exc_info=True)
        return []
