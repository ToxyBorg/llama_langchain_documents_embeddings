import json
import os
from dotenv import load_dotenv

from langchain import FAISS
from langchain.embeddings import LlamaCppEmbeddings
from loading_embeddings import load_embeddings
from save_vectorstore import save_vectorstore


def create_vectorstore_from_json(json_files_paths: list[str], model_path):
    # Load embeddings
    embeddings = LlamaCppEmbeddings(model_path=model_path)

    saving_embeddings_file_name: str = os.getenv("SAVING_EMBEDDINGS_FILE_NAME")
    embeddings_path = os.path.join(
        os.getcwd(), "embeddings_data/"+saving_embeddings_file_name+".pkl")
    text_embeddings = load_embeddings(embeddings_path)

    for json_file_path in json_files_paths:
        with open(json_file_path, "r") as f:
            documents = json.load(f)
        texts = [doc["page_content"] for doc in documents]

    text_embedding_pairs = list(zip(texts, text_embeddings))
    faiss = FAISS.from_embeddings(text_embedding_pairs, embeddings)

    return faiss


"""################# CALLING THE FUNCTION #################"""
load_dotenv()  # Load environment variables from .env file

path_to_ggml_model: str = os.getenv("PATH_TO_LLAMA_CPP_GGML")

documents_chunks_json_list: list[str] = json.loads(
    os.getenv("LIST_OF_DOCUMENTS_JSON_CHUNKS"))

vectorstore = create_vectorstore_from_json(
    json_files_paths=documents_chunks_json_list, model_path=path_to_ggml_model)

saving_vectorstore_file_name: str = os.getenv("SAVING_VECTORSTORE_FILE_NAME")
save_vectorstore(vectorstore=vectorstore,
                 file_name=saving_vectorstore_file_name)
