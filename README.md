# Document Indexing and Query Engine

This project indexes documents from a specified directory and its subdirectories for use with a query engine. It supports embedding documents using a specified language model and provides an interactive query interface.

## Features

- Recursively finds and processes all directories within a specified directory.
- Loads and indexes documents from these directories.
- Uses a specified language model to embed the documents.
- Builds an index from the embedded documents.
- Creates a query engine from the index.
- Interactive query loop to search the indexed documents.
- Progress bar to indicate the progress of loading documents from directories.

## Prerequisites

- Python 3.12 or higher
- The following Python libraries: see below

You can setup the virtual environment using venv:

```sh
python3 -m venv llamaragenv
source llamaragenv/bin/activate
python -m ensurepip --upgrade
pip install --upgrade pip
```

You can install the required libraries using pip:
```sh
pip install tqdm
pip install ollama
pip install llama-index-core
pip install llama-index-llms-openai
pip install llama-index-llms-replicate
pip install llama-index-embeddings-huggingface
pip install llama-index-llms-ollama
pip install llama-index-readers-file
```

## Usage
Command-Line Arguments:

* data_folder (required): The folder containing documents to be indexed. The script will process all subdirectories within this folder.
* --model (optional): The language model to use for embeddings. The default model is 'llama3'.

### Example
Using the default model:

```sh
python script.py /path/to/your/data
```
Specifying a different model:

```sh
python script.py /path/to/your/data --model different_model
```

## Interactive Query Loop
Enter your query to search the indexed documents. Type 'exit' to terminate the loop and end the script.

## How It Works
- Directory Traversal: The script recursively finds all directories within the specified directory.
- Document Loading: It loads documents from these directories using a document reader.
- Embedding: The specified language model is used to embed the documents.
- Indexing: An index is built from the embedded documents.
- Query Engine: A query engine is created from the index.
- Query Interface: The user can enter queries to search the indexed documents interactively.

### Progress Display
During the indexing process, the script displays a progress bar to indicate the progress of loading documents from directories.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- llama_index
- HuggingFace
tqdm
