import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings

# PDFS_PATH = "C:\\Users\\gfrag\\Desktop\\Workspace\\Fairytales"
PDFS_PATH = "C:\\Users\\gfrag\\Desktop\\Workspace\\StarWarsScripts"
CHROMA_PATH = "chroma"

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="llama3.1:latest")
    return embeddings

def load_documents():
    '''Load documents on which we will be asking questions'''
    document_loader = PyPDFDirectoryLoader(PDFS_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    '''Splitting pdfs into chunks to create a better DB'''    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)



#In order to not create the db from scratch every time the script runs
#we use this function to id each pdf and pdf page so that we are able 
# to add to the db instead of re-initializing it
def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        curr_page_id = f"{source}:{page}"
        
        # If the page ID is the same as the last one, increment the index.
        if curr_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
            
        # Calculate the chunk ID.
        chunk_id = f"{curr_page_id}:{current_chunk_index}"
        last_page_id = curr_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id
           
    return chunks



def add_to_chroma(chunks: list[Document]):
    # Load or create the database
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    
    # Calculate Page IDs for all chunks
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # get the existing documents
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    
    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist() # happens automatically
    else:
        print("✅ No new documents to add")
    
    
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    
    
    
if __name__ == "__main__":
    main()