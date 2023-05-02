"""
    This code defines a function called using_vectorstore_with_llama that takes in three arguments: 
        model_path, 
        path_to_vectorstore, and 
        query.

    Within the function, it first creates an instance of the LlamaCppEmbeddings class from the langchain.embeddings module using the model_path argument. It then uses this instance to embed the query text.

    Next, it loads a FAISS vectorstore using the FAISS.load_local method from the langchain module, passing in the path_to_vectorstore argument and the llama instance.

    Finally, it uses the faiss.similarity_search_by_vector method to find the most similar documents to the query, and returns a list of tuples containing the document IDs and their similarity scores.
"""


import os
from typing import List
from dotenv import load_dotenv

from langchain.embeddings import LlamaCppEmbeddings
from langchain import FAISS, LLMChain, LlamaCpp
from langchain.schema import Document

from langchain.prompts import PromptTemplate
from pypdf import DocumentInformation


def using_vectorstore_with_llama(model_path: str, path_to_vectorstore: str, query: str):
    # Embed the query text
    llama = LlamaCppEmbeddings(model_path=model_path)
    query_embedding = llama.embed_query(query)

    # Load the FAISS vectorstore
    faiss = FAISS.load_local(path_to_vectorstore, llama)

    # Find the most similar documents to the query
    docs_and_scores = faiss.similarity_search_by_vector(embedding=query_embedding, k=2)

    return docs_and_scores


def using_vectorstore_with_llama_and_corpus(
    model_path: str, corpus: List[Document], query: str
) -> str:
    # Use the Llama GGML model to answer the query
    llm = LlamaCpp(model_path=model_path, n_ctx=1024)
    template = """ \
        Given the following chunks of text, try to answer the query given to the best you can.
        Only use the data provided in the Paragraphs. 
        If you cannot deduce any useful information from the Paragraphs to answer the query, DO say so.
        
        Paragraphs: 
        
        {Paragraphs}
        
        Query: {Query}
        
        Answer: """

    prompt = PromptTemplate(template=template, input_variables=["Paragraphs", "Query"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    Paragraphs = ""
    for doc_data, metadata in corpus:
        doc_text: str = doc_data[1]

        Paragraphs += "     " + doc_text

    answer = llm_chain.run(Query=query, Paragraphs=Paragraphs)

    return answer


"""################# CALLING THE FUNCTION #################"""

load_dotenv()  # Load environment variables from .env file

path_to_ggml_model: str = os.getenv("PATH_TO_LLAMA_CPP_GGML")

saving_vectorstore_file_name: str = os.getenv("SAVING_VECTORSTORE_FILE_NAME")
saving_vectorstore_directory: str = os.getenv("SAVING_VECTORSTORE_DIRECTORY")
vectorstore_directory_path = os.path.join(os.getcwd(), saving_vectorstore_directory)
vectorstore_path = os.path.join(
    vectorstore_directory_path, saving_vectorstore_file_name + ".faiss"
)

query = "tell me more about the commonwealth spelling rules"


docs_and_scores = using_vectorstore_with_llama(
    model_path=path_to_ggml_model, path_to_vectorstore=vectorstore_path, query=query
)


# # Pass the relevant documents to the Llama GGML model for further processing
# llm = LlamaCpp(model_path=path_to_ggml_model)

# template = "Given the following , \
#     it is your job to write a synopsis for that title. Title: {title} Playwright: This is a synopsis for the above play:"
# prompt_template = PromptTemplate(input_variables=["title"], template=template)
# synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, callback_manager=manager)


# llm_outputs = synopsis_chain.run(answer, query)

# print("\n")
# print(llm_outputs)
# print("\n")

answer = using_vectorstore_with_llama_and_corpus(
    corpus=docs_and_scores, model_path=path_to_ggml_model, query=query
)

print("\n")
print(answer)
