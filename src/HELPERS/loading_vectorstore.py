import os
from dotenv import load_dotenv

from langchain import FAISS
from langchain.embeddings import LlamaCppEmbeddings

"""################# LOADING THE FAISS VECTORSTORE #################"""


def load_faiss_vectorstore(file_path: str, model_path: str):
    # Load LlamaCppEmbeddings object
    llama = LlamaCppEmbeddings(model_path=model_path)

    return FAISS.load_local(file_path, llama)


"""################# CALLING THE FUNCTION #################"""

load_dotenv()  # Load environment variables from .env file

path_to_ggml_model: str = os.getenv("PATH_TO_LLAMA_CPP_GGML")

saving_vectorstore_file_name: str = os.getenv("SAVING_VECTORSTORE_FILE_NAME")
vectorstore_path = os.path.join(
    os.getcwd(), "vectorstore_data/"+saving_vectorstore_file_name+".faiss")

vectorstore = load_faiss_vectorstore(file_path=vectorstore_path,
                                     model_path=path_to_ggml_model)
