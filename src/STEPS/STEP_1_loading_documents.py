"""
    This function takes in a docs_directory_path parameter, 
    which is the path to the directory containing the unstructured documents 
    to be loaded and split into smaller chunks.

    The function then iterates through all the files in the directory 
    using os.listdir(docs_directory_path). 
    For each file, it loads the unstructured documents from 
    the file path using UnstructuredFileLoader and loader.load().

    Next, the function splits the documents into smaller chunks 
    using RecursiveCharacterTextSplitter. 
    The chunk_size parameter specifies the maximum number of 
    characters in each chunk, while the chunk_overlap parameter 
    specifies the number of characters to overlap between adjacent chunks. 
    The length_function parameter specifies the function to use 
    to calculate the length of each document.

    The function then creates a list of dictionaries, where each 
    dictionary contains a single chunk of a document. 
    The page_content key in each dictionary contains 
    the text content of the chunk.

    The function then creates a directory for the chunked 
    data if it doesn't already exist, using the os.makedirs() function. 
    The directory path is specified 
    by the DIRECTORY_FOR_DOCUMENTS_JSON_CHUNKS environment variable, 
    which is loaded using os.getenv().

    Finally, the function saves the chunked documents to a JSON file
    with a dynamic name. The file name is generated by removing the 
    file extension from the original file name using os.path.splitext(file_name)[0], 
    and appending _chunks.json. The json.dump() function is 
    used to write the documents to the file.
"""

import os
import json
from dotenv import load_dotenv

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


"""
    Load unstructured documents from a directory path, 
    split them into smaller chunks, and 
    save them as JSON files in a specified directory.

    Args:
        docs_directory_path (str): The path to the directory containing the unstructured documents.
        save_json_chunks_directory (str): The path to the directory where the JSON files 
        containing the smaller document chunks will be saved.

    Returns:
        None
"""


def load_documents(docs_directory_path: str, save_json_chunks_directory: str):
    # iterating through all the files in the directory
    for file_name in os.listdir(docs_directory_path):
        file_path = os.path.join(docs_directory_path, file_name)
        # Load unstructured documents from file path
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, length_function=len
        )
        documents = []
        for doc in docs:
            chunks = text_splitter.create_documents([doc.page_content])
            for chunk in chunks:
                documents.append({"page_content": chunk.page_content})

        # Create directory for chunked data if it doesn't exist
        if not os.path.exists(save_json_chunks_directory):
            os.makedirs(save_json_chunks_directory)

        # Save documents to JSON file with dynamic name
        file_name = os.path.splitext(file_name)[0]
        json_file_path = os.path.join(
            save_json_chunks_directory, f"{file_name}_chunks.json"
        )
        with open(json_file_path, "w") as f:
            json.dump(documents, f)


"""################# CALLING THE FUNCTION #################"""

print("\n####################### LOADING DOCUMENTS ########################\n")

load_dotenv()  # Load environment variables from .env file

docs_directory_path = os.getenv("DIRECTORY_DOCUMENTS_TO_LOAD")

save_json_chunks_directory = os.getenv("DIRECTORY_FOR_DOCUMENTS_JSON_CHUNKS")

# Load documents and save them
load_documents(
    docs_directory_path=docs_directory_path,
    save_json_chunks_directory=save_json_chunks_directory,
)

print("\n####################### LOADED AND SAVED ########################\n")
