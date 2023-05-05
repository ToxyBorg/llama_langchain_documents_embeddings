# # LLaMA x Langchain documents embeddings
Just testing langchain's documents embeddings with llamacpp

# # STEP 1 LOADING AND CHUNKING THE DOCUMENTS:

  ## The function load_documents:
    Loads unstructured documents from a directory path, splits them into smaller chunks, and returns a list of objects. 
    Each object has two properties: the name of the document that was chunked, and the chunked data itself. 
    The function uses the UnstructuredFileLoader or PyPDFLoader class from the langchain.document_loaders module to 
    load the documents from the directory path, and the RecursiveCharacterTextSplitter class from the 
    langchain.text_splitter module to split the documents into smaller chunks. 
    
    The resulting list of objects is returned by the function.
    
  ## The function save_documents:
    Saves a list of objects to JSON files. 
    Each object in the list should have two properties: the name of the document that was chunked, and the chunked data itself. 
    The JSON file should be named after the document name, with "Chunks" appended to the end of the name. 
    The content of the JSON file should be the chunked data. 
    The function uses the os and json modules to create the directory for the chunked data if it doesn't exist, 
    and to save the documents to JSON files with dynamic names. 
    
    The resulting JSON files are saved in the directory specified by the save_json_chunks_directory argument

# # STEP 2 CREATING AND SAVING THE EMBEDDINGS:

  ## The function create_embeddings:
    Loads the LlamaCppEmbeddings model using the provided path, and 
    then iterates through each JSON file in the directory specified by load_json_chunks_directory. 
    For each file, it extracts the text content from the JSON and passes it to 
    the LlamaCppEmbeddings model to generate embeddings. 
    
    The embeddings are then added to a list, which is returned by the function.

  ## The function save_embeddings:
    The function first creates a directory at the specified path if it does not already exist. 
    It then creates a file path by joining the directory path and file name with a ".pkl" extension. 
    
    Finally, it saves the embeddings object to the binary file using the "pickle" module

# # STEP 3 CREATING AND SAVING VECTORSTORES:

  ## The function create_vectorstore_from_json:
    Creates a FAISS index from text embeddings extracted from JSON files in a specified directory.
    The function loads the embeddings, reads the JSON files, extracts the text values, creates text 
    embedding pairs, and 
    
    Returns a FAISS index from the pairs.
    
  ## The function load_embeddings:
    Loads embeddings from a file using the pickle module.
    The function opens the file in binary mode, loads the embeddings using pickle.load(), and 
        
    Returns the embeddings.
    
  ## The function save_vectorstore:
    Saves a FAISS index as a file at the specified directory path and file name.
    The function creates the directory if it doesn't exist, creates the file path. 
    
    Finally, it saves the FAISS index to the file.

# # STEP 4 USING THE CREATED VECTOR STORE FROM EMBEDDINGS TO QUERY THE DOCS

  ## using_vectorstore_similarity_search: 
    This function takes in :
      - a path to a pre-trained language model, 
      - a path to a vector store, and 
      - a query string. 
    It first embeds the query text using the pre-trained language model, 
    then loads the vector store using the FAISS library. 
    
    Finally, it uses the vector store to find the k most similar documents to the query, where k is set to 4 in this implementation. 
    The function returns a list of Document objects, where each Document represents one of the most similar documents to the query.

  ## Q_and_A_implementation: 
    This function takes in:
      - a path to a pre-trained language model, 
      - a list of Document objects representing the most similar documents to a query, and 
      - the query string itself. 
    It loads a pre-trained question-answering model using the load_qa_chain function from the 
    langchain.chains.question_answering module, and applies this model to the list of Document objects and 
    the query string to generate an answer. 
    
    The function returns the answer as a string.

  ## Finally
    The code then loads environment variables from a .env file, 
    sets up the paths to the pre-trained language model and the vector store, and 
    defines the query string. 

    It calls using_vectorstore_similarity_search to find the most similar documents to the query, and 
    then calls Q_and_A_implementation to generate an answer to the query using the pre-trained question-answering model. 
    Finally, it prints the answer to the console.




