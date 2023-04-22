import os
from dotenv import load_dotenv

from loading_embeddings import load_embeddings
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import Chroma

"""################# USING THE EMBEDDINGS FOR SEMANTIC SEARCH #################"""


def get_embeddings_for_question(embeddings_path: str, model_path: str, question: str):
    # Load embeddings from pickle file
    embeddings = load_embeddings(embeddings_path)

    # Create the vectorstore
    state_of_union_store = Chroma.from_texts(
        embeddings, collection_name="state-of-union")


"""################# CALLING THE FUNCTION #################"""
load_dotenv()  # Load environment variables from .env file

path_to_ggml_model: str = os.getenv("PATH_TO_LLAMA_CPP_GGML")

saving_embeddings_file_name: str = os.getenv("SAVING_EMBEDDINGS_FILE_NAME")
embeddings_path = os.path.join(
    os.getcwd(), "embeddings_data/"+saving_embeddings_file_name+".pkl")

question = "what should I do if people are speaking in the background but the file's main speakers have no interaction with them"


answer = get_embeddings_for_question(
    embeddings_path=embeddings_path, model_path=path_to_ggml_model, question=question)


print(answer)
