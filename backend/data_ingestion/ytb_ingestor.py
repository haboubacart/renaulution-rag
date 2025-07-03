from utils import clean_raw_text, join_short_lines, create_chunks
from langchain_community.document_loaders import YoutubeLoader

def extract_and_split_ytb(url: str, chunk_size=450, chunk_overlap=64):
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=False,   
        language=["fr", "en"]  
    )
    documents = loader.load()
    all_ytb_chunks = create_chunks(documents, "ytb", chunk_size, chunk_overlap, False)

    return all_ytb_chunks