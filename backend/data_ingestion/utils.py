from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import uuid



def clean_raw_text(text: str) -> str:
    # 1. Fusionner les lignes coupées
    text = re.sub(r'(?<![.\n])\n(?![\n0-9•\-])', ' ', text)
    
    # 2. Supprimer les pieds de page / mentions AMF
    text = re.sub(r"GROUPE RENAULT\s+I\s+DOCUMENT D.*?\d{4}", '', text, flags=re.IGNORECASE)
    text = re.sub(r"Ce document.*?AMF", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 3. Supprimer les lignes parasites du sommaire
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if re.match(r'^\d{1,3}$', line): continue
        if re.match(r'^[A-Z\s\-]{5,}$', line): continue
        if re.match(r'^\d+(\.\d+)*\s+', line): continue
        clean_lines.append(line)
    text = "\n".join(clean_lines)

    # 4. Supprimer les numéros de page
    text = re.sub(r'^Page \d+.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)

    # 5. Nettoyage espaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)

    return text.strip()

def join_short_lines(text: str, min_len: int = 60) -> str:
    lines = text.splitlines()
    result = []
    buffer = ""
    for i, line in enumerate(lines):
        line = line.strip()
        # Ignore ligne vide
        if not line:
            if buffer:
                result.append(buffer.strip())
                buffer = ""
            result.append("")
            continue
        # Si ligne est courte et ne finit pas par ponctuation forte
        if len(line) < min_len and not re.search(r'[.!?:;»”]\s*$', line):
            buffer += " " + line  # On accumule
        else:
            buffer += " " + line
            result.append(buffer.strip())
            buffer = ""
    # Append dernier buffer si nécessaire
    if buffer:
        result.append(buffer.strip())

    return "\n".join(result)

def create_chunks(documents : Document, filename : str, chunk_size : int, chunk_overlap : int, activate_cleaning = True) -> list :
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    all_chunks = []
    for doc in documents:
        if activate_cleaning : 
            cleaned_text_page = join_short_lines(clean_raw_text(doc.page_content))
            chunks = splitter.split_text(cleaned_text_page)
        else :
            chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            all_chunks.append(Document(
                page_content = chunk,
                metadata={
                    "source": filename,
                    "id": str(uuid.uuid4())
                }
            ))
    return all_chunks