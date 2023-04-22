# # LLaMA x Langchain documents embeddings
Just testing langchain's documents embeddings with llamacpp

# # STEP 1 LOADING AND CHUNKING THE DOCUMENTS:

  ## The function load_documents:
    Loads unstructured documents from a directory path, 
    splits them into smaller chunks, and saves them as JSON files with dynamic names in a specified directory. 
    It uses the UnstructuredFileLoader and RecursiveCharacterTextSplitter classes from the langchain.document_loaders 
    and langchain.text_splitter modules, respectively.
  
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
