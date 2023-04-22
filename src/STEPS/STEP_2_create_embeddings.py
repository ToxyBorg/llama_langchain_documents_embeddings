import json
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
import os

from save_embeddings import save_embeddings

"""################# CREATING EMBEDDINGS #################"""


# def create_embeddings(json_files_paths: list[str], model_path: str):
#     # Load LlamaCppEmbeddings object
#     embeddings = LlamaCppEmbeddings(model_path=model_path)

#     # Embed text from JSON files using LlamaCppEmbeddings
#     all_embeddings: Embeddings = []

#     for json_file_path in json_files_paths:
#         with open(json_file_path, "r") as f:
#             documents = json.load(f)

#         texts = [doc["page_content"] for doc in documents]
#         embeddings_list = embeddings.embed_documents(texts)
#         all_embeddings.extend(embeddings_list)

#     return all_embeddings

def create_embeddings(json_dir_path: str, model_path: str):
    # Load LlamaCppEmbeddings object
    embeddings = LlamaCppEmbeddings(model_path=model_path)

    # Embed text from JSON files in directory using LlamaCppEmbeddings
    all_embeddings: Embeddings = []

    for filename in os.listdir(json_dir_path):
        if filename.endswith(".json"):
            with open(os.path.join(json_dir_path, filename), "r") as f:
                documents = json.load(f)

            texts = [doc["page_content"] for doc in documents]
            embeddings_list = embeddings.embed_documents(texts)
            all_embeddings.extend(embeddings_list)

    return all_embeddings


"""################# CALLING THE FUNCTION #################"""

load_dotenv()  # Load environment variables from .env file

# # Create embeddings
# documents_chunks_json_list: list[str] = json.loads(
#     os.getenv("LIST_OF_DOCUMENTS_JSON_CHUNKS"))

documents_chunks_json_list = os.getenv("DIRECTORY_FOR_DOCUMENTS_JSON_CHUNKS")

path_to_ggml_model: str = os.getenv("PATH_TO_LLAMA_CPP_GGML")

saving_embeddings_file_name: str = os.getenv("SAVING_EMBEDDINGS_FILE_NAME")

embeddings = create_embeddings(
    json_files_paths=documents_chunks_json_list, model_path=path_to_ggml_model)

# Save embeddings
save_embeddings(embeddings=embeddings, file_name=saving_embeddings_file_name)
