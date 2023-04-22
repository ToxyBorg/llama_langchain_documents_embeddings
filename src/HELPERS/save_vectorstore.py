import os
from langchain.vectorstores.faiss import FAISS


"""################# SAVE THE CREATED VECTORSTORE #################"""


def save_vectorstore(vectorstore: FAISS, file_name: str):

    directory = os.path.join(os.getcwd(), "vectorstore_data")
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name+'.faiss')

    vectorstore.save_local(file_path)
