from pathlib import Path
from dotenv import load_dotenv
import logging
import os
from getpass import getpass
from langchain_community.document_loaders import S3DirectoryLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import shutil

# Load environment variables
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT_DIR = Path.home()/'.ragbot'
VECTOR_DB_DIR = ROOT_DIR/"vector_dbs"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
CONTEXT_DIR = ROOT_DIR/"context_folder"
CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

def get_aws_config():
    aws_config = {}
    while not aws_config['region_name']:
        aws_config['region_name'] = os.getenv('AWS_REGION_NAME') or input('Enter AWS Region Name: ')
        if not aws_config['region_name']:
            logger.info("Region name can not be blank")
    
    while not aws_config['aws_access_key_id']:        
        aws_config['aws_access_key_id'] = os.getenv('AWS_ACCESS_KEY') or getpass('Enter AWS Access Key: ')
        if not aws_config['aws_access_key_id']:
            logger.info("AWS Access Key can not be blank")
            
    while not aws_config['aws_secret_access_key']: 
        aws_config['aws_secret_access_key'] = os.getenv('AWS_SECRET_ACCESS_KEY') or getpass('Enter AWS Secret Access Key: ')
        if not aws_config['aws_secret_access_key']:
            logger.info("AWS Secret Access Key can not be blank")
    return aws_config

def load_documents():
    source_type = input("Enter source type (local/s3): ").lower()
    if source_type == "local":
        while True:
            folder_name = input('Enter the folder name containing PDF files: ')
            if (CONTEXT_DIR/folder_name).exists():
                loader = DirectoryLoader(str(CONTEXT_DIR/folder_name), glob="**/*.pdf", loader_cls=PyPDFLoader)
                documents = loader.load()
                if len(documents) >= 1:
                    break
                else:
                    logger.warning(f"No PDF documents found in directory - {CONTEXT_DIR/folder_name}")
            else:
                logger.warning(f"Unable to find {CONTEXT_DIR/folder_name}. Please create the folder or provide different folder name.")
    if source_type == "s3":
        while True:
            bucket_name = input('Enter the bucket name containing PDF files: ')
            aws_config = get_aws_config()
            loader = S3DirectoryLoader(bucket_name, **aws_config)
            documents = loader.load()
            if len(documents) >= 1:
                break
            else:
                logger.warning(f"No PDF documents found in directory - {CONTEXT_DIR/folder_name}")
    logger.info(f"{len(documents)} documents found.")
    return documents

def create_vector_db(documents, db_name):
    logger.info("Creating vector database...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    text_splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="standard_deviation")
    chunks = text_splitter.split_documents(documents)
    logger.info(f"{len(chunks)} chunks created from {len(documents)} documents.")

    vector_db = FAISS.from_documents(chunks, embedding_model)

    db_path = VECTOR_DB_DIR / db_name
    vector_db.save_local(str(db_path))
    logger.info(f"Vector database '{db_name}' created successfully.")
    
def resync_vector_db(db_name):
    db_path = VECTOR_DB_DIR / db_name
    if not db_path.exists():
        logger.error(f"Vector database '{db_name}' does not exist.")
        return

    documents = load_documents()

    logger.info(f"Deleting existing vector database '{db_name}'...")
    shutil.rmtree(db_path)

    create_vector_db(documents, db_name)
    logger.info(f"Vector database '{db_name}' resynced successfully.")
    
def delete_vector_db(db_name):
    db_path = VECTOR_DB_DIR / db_name
    if db_path.exists():
        shutil.rmtree(db_path)
        logger.info(f"Vector database '{db_name}' deleted successfully.")
    else:
        logger.error(f"Vector database '{db_name}' does not exist.")
        
def main():
    while True:
        print("\nVector Database Management")
        print("1. Create new vector database")
        print("2. Resync existing vector database")
        print("3. Delete vector database")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            db_name = input("Enter a name for the new vector database: ")
            db_path = VECTOR_DB_DIR / db_name
            if db_path.exists():
                resync = input(f"Vector database '{db_name}' already exists. Do you want to resync it? (y/n): ").lower()
                if resync == 'y':
                    resync_vector_db(db_name)
                else:
                    continue
            else:
                documents = load_documents()
                create_vector_db(documents, db_name)

        elif choice == "2":
            db_name = input("Enter the name of the vector database to resync: ")
            resync_vector_db(db_name)

        elif choice == "3":
            db_name = input("Enter the name of the vector database to delete: ")
            delete_vector_db(db_name)

        elif choice == "4":
            logger.info("Exiting the program.")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
    