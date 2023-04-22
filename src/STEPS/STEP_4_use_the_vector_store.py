import os
from dotenv import load_dotenv

from langchain.embeddings import LlamaCppEmbeddings
from langchain import FAISS

"""################# USING THE FAISS VECTORSTORE #################"""


def using_vectorstore_with_llama(model_path: str, path_to_vectorstore: str, query: str):

    # Embed the query text
    llama = LlamaCppEmbeddings(model_path=model_path)
    query_embedding = llama.embed_query(query)

    # Load the FAISS vectorstore
    faiss = FAISS.load_local(path_to_vectorstore, llama)

    # Find the most similar documents to the query
    docs_and_scores = faiss.similarity_search_by_vector(query_embedding, k=1)

    return docs_and_scores


"""################# CALLING THE FUNCTION #################"""

load_dotenv()  # Load environment variables from .env file

path_to_ggml_model: str = os.getenv("PATH_TO_LLAMA_CPP_GGML")

saving_vectorstore_file_name: str = os.getenv("SAVING_VECTORSTORE_FILE_NAME")
vectorstore_path = os.path.join(
    os.getcwd(), "vectorstore_data/"+saving_vectorstore_file_name+".faiss")

query = "How do i monitor activities in the zkr systems"


answer = using_vectorstore_with_llama(
    model_path=path_to_ggml_model, path_to_vectorstore=vectorstore_path, query=query)


print(answer)
