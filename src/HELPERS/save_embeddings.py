import pickle
import os

from langchain.embeddings.base import Embeddings


"""################# SAVE THE CREATED EMBEDDINGS #################"""


def save_embeddings(embeddings: Embeddings, file_name: str):

    directory = os.path.join(os.getcwd(), "embeddings_data")
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name+'.pkl')

    # Save embeddings to binary file
    with open(file_path, "wb") as f:
        pickle.dump(embeddings, f)
