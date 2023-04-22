import pickle

from langchain.embeddings.base import Embeddings


def load_embeddings(file_path: str) -> Embeddings:
    with open(file_path, "rb") as f:
        embeddings: Embeddings = pickle.load(f)

    return embeddings
