from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL, DEVICE


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': DEVICE}
    )