"""
This script indexes documents from a specified directory and its subdirectories for use with a query engine.

The script performs the following steps:
1. Recursively finds all directories within a specified directory.
2. Loads documents from these directories using a document reader.
3. Uses a specified language model to embed the documents.
4. Builds an index from the embedded documents.
5. Creates a query engine from the index.
6. Enters an interactive loop where the user can enter queries to search the indexed documents.

The script supports the following command-line arguments:
- data_folder (required): The folder containing documents to be indexed. The script will process all subdirectories within this folder.
- --model (optional): The language model to use for embeddings. The default model is 'llama3'.

During the indexing process, the script displays a progress bar to indicate the progress of loading documents from directories.

Example usage:
    python script.py /path/to/your/data
    python script.py /path/to/your/data --model different_model

Prerequisites:
- Python 3.12 or higher
- ollama library: You need to have the ollama library installed.
- The models you want to use should be installed and available to the script.
- tqdm library: To display progress bars.
- llama_index library: For core indexing functionalities and embeddings.

Modules used:
- os: To navigate through directories.
- argparse: To handle command-line arguments.
- concurrent.futures: To enable parallel processing of directories.
- tqdm: To display a progress bar.
- llama_index.core: For core indexing functionalities.
- llama_index.embeddings.huggingface: For embedding documents using the HuggingFace library.
- llama_index.llms.ollama: For setting up the language model.

Functions:
- get_all_directories(directory): Recursively finds all directories within the specified directory.
- load_documents_from_single_directory(directory): Loads documents from a single directory.
- load_documents_from_directory(directory): Loads documents from all directories within the specified directory, using parallel processing.
- main(data_folder, model_name): Main function to load documents, build the index, and run the interactive query loop.

Interactive Loop:
- The user can enter queries to search the indexed documents.
- Typing 'exit' will terminate the loop and end the script.
"""

import os
import argparse
import concurrent.futures
from tqdm import tqdm
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Function to recursively find directories in a directory
def get_all_directories(directory):
    directories = []
    for root, dirs, _ in os.walk(directory):
        for dir in dirs:
            directories.append(os.path.join(root, dir))
    return directories

# Load documents from a single directory
def load_documents_from_single_directory(directory):
    return SimpleDirectoryReader(directory).load_data()

# Load documents from all directories in the directory and subdirectories
def load_documents_from_directory(directory):
    dir_paths = get_all_directories(directory)
    documents = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_dir = {executor.submit(load_documents_from_single_directory, dir_path): dir_path for dir_path in dir_paths}
        for future in tqdm(concurrent.futures.as_completed(future_to_dir), total=len(future_to_dir), desc="Indexing documents"):
            try:
                documents.extend(future.result())
            except Exception as e:
                print(f"Error loading documents from directory {future_to_dir[future]}: {e}")
    return documents

def main(data_folder, model_name):
    # Load documents
    documents = load_documents_from_directory(data_folder)

    # bge-base embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # ollama
    Settings.llm = Ollama(model=model_name, request_timeout=360.0)

    # Build index
    index = VectorStoreIndex.from_documents(documents)

    # Create query engine
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=5)

    # Interactive loop
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        streaming_response = query_engine.query(query)
        streaming_response.print_response_stream()
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('data_folder', type=str, help='The folder to use for embeddings')
    parser.add_argument('--model', type=str, default='llama3', help='The model to use for embeddings (default: llama3)')

    args = parser.parse_args()
    main(args.data_folder, args.model)
