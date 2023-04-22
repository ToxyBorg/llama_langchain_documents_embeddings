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
