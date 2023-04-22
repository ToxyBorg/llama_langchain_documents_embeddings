"""
    This code defines a function called load_documents that loads unstructured documents from a directory path, splits them into smaller chunks, and returns a list of objects. 
    
    Each object has two properties: 
    the name of the document that was chunked, and the chunked data itself. 
    
    The function uses the UnstructuredFileLoader or PyPDFLoader class from the langchain.document_loaders module to load the documents from the directory path, and the RecursiveCharacterTextSplitter class from the 
    langchain.text_splitter module to split the documents into smaller chunks. 
    
    The resulting list of objects is returned by the function.
"""

import os
import sys
from dotenv import load_dotenv

from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from typing import List, Dict, Union

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from HELPERS.save_chunked_docs import save_documents


"""
    Loads unstructured documents from a directory path, splits them into smaller chunks, 
    and returns a list of objects.
    Each object has two properties: 
        the name of the document that was chunked, 
        and the chunked data itself.
    
    Args:
    - docs_directory_path (str): The path to the directory containing the documents to be chunked.
    
    Returns:
    - List[Dict[str, Union[str, List[Dict[str, str]]]]]: A list of objects, 
        where each object has two properties:
            the name of the document that was chunked, 
            and the chunked data itself.
"""


def load_documents(
    docs_directory_path: str,
) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
    result = []
    # iterating through all the files in the directory
    for file_name in os.listdir(docs_directory_path):
        file_path = os.path.join(docs_directory_path, file_name)
        # Load unstructured documents from file path

        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path=file_path)
        else:
            loader = UnstructuredFileLoader(file_path=file_path)

        docs = loader.load()

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, length_function=len
        )
        chunks = []
        for doc in docs:
            chunks.extend(text_splitter.create_documents([doc.page_content]))

        chunks = [
            {"chunk_" + str(i + 1): chunk.page_content}
            for i, chunk in enumerate(chunks)
        ]

        # Add document name and chunked data to result list
        file_name = os.path.splitext(file_name)[0]
        result.append({"name": file_name, "chunks": chunks})

    return result


"""################# CALLING THE FUNCTION #################"""

print("\n####################### LOADING DOCUMENTS ########################\n")

load_dotenv()  # Load environment variables from .env file

docs_directory_path = os.getenv("DIRECTORY_DOCUMENTS_TO_LOAD")

# Load documents
loaded_and_chunked_docs = load_documents(docs_directory_path=docs_directory_path)

print("\n####################### DOCUMENTS LOADED ########################\n")


print("\n####################### DOCUMENT CHUNKS LOADED ########################\n")

save_json_chunks_directory = os.getenv("DIRECTORY_FOR_DOCUMENTS_JSON_CHUNKS")

# Save documents
save_documents(
    documents=loaded_and_chunked_docs,
    save_json_chunks_directory=save_json_chunks_directory,
)


print("\n####################### DOCUMENT CHUNKS SAVED ########################\n")
