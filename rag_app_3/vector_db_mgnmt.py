import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import logging
import os
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

# Custom CSS to improve aesthetics
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .stSelectbox {
        margin-bottom: 20px;
    }
    .stTextInput {
        margin-bottom: 20px;
    }
    .stAlert {
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def get_aws_config():
    aws_config = {}
    aws_config['region_name'] = os.getenv('AWS_REGION_NAME') or st.text_input('Enter AWS Region Name:')
    if not aws_config['region_name']:
        st.error('Invalid Region Name.')
        st.stop()
    aws_config['aws_access_key_id'] = os.getenv('AWS_ACCESS_KEY') or st.text_input('Enter AWS Access Key:', type='password')
    if not aws_config['aws_access_key_id']:
        st.error('Invalid AWS Access Key ID.')
        st.stop()
    aws_config['aws_secret_access_key'] = os.getenv('AWS_SECRET_ACCESS_KEY') or st.text_input('Enter AWS Secret Access Key:', type='password')
    if not aws_config['aws_secret_access_key']:
        st.error('Invalid AWS Secret Access Key ID.')
        st.stop()
    return aws_config

def load_documents(source_type, folder_name=None, bucket_name=None):
    if source_type == "local":
        if (CONTEXT_DIR/folder_name).exists():
            loader = DirectoryLoader(str(CONTEXT_DIR/folder_name), glob="**/*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            if len(documents) >= 1:
                return documents
            else:
                st.warning(f"No PDF documents found in directory - {CONTEXT_DIR/folder_name}")
        else:
            st.warning(f"Unable to find {CONTEXT_DIR/folder_name}. Please create the folder or provide different folder name.")
    elif source_type == "s3":
        aws_config = get_aws_config()
        loader = S3DirectoryLoader(bucket_name, **aws_config)
        documents = loader.load()
        if len(documents) >= 1:
            return documents
        else:
            st.warning(f"No PDF documents found in S3 bucket - {bucket_name}")
    return None

def create_vector_db(documents, db_name):
    st.info("Creating vector database...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    text_splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="standard_deviation")
    chunks = text_splitter.split_documents(documents)
    st.info(f"{len(chunks)} chunks created from {len(documents)} documents.")

    vector_db = FAISS.from_documents(chunks, embedding_model)

    db_path = VECTOR_DB_DIR / db_name
    vector_db.save_local(str(db_path))
    st.success(f"Vector database '{db_name}' created successfully.")

def resync_vector_db(db_name, documents):
    db_path = VECTOR_DB_DIR / db_name
    if not db_path.exists():
        st.error(f"Vector database '{db_name}' does not exist.")
        return

    st.info(f"Deleting existing vector database '{db_name}'...")
    shutil.rmtree(db_path)

    create_vector_db(documents, db_name)
    st.success(f"Vector database '{db_name}' resynced successfully.")

def delete_vector_db(db_name):
    db_path = VECTOR_DB_DIR / db_name
    if db_path.exists():
        shutil.rmtree(db_path)
        st.success(f"Vector database '{db_name}' deleted successfully.")
    else:
        st.error(f"Vector database '{db_name}' does not exist.")

def main():
    st.title("📚 Vector Database Manager")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Actions")
        action = st.radio("Choose an action", 
                          ["Create new vector database", 
                           "Resync existing vector database", 
                           "Delete vector database"])

    with col2:
        st.subheader("Configuration")
        if action in ["Create new vector database", "Resync existing vector database"]:
            if action == "Resync existing vector database":
                vector_databases = [x.name for x in VECTOR_DB_DIR.iterdir() if x.is_dir()]
                db_name = st.selectbox("Select the vector database to resync:", vector_databases)
            if action == "Create new vector database":
                db_name = st.text_input("Enter the name for the new vector database:")

            source_type = st.selectbox("Select source type", ["local", "s3"])
            if source_type == "local":
                context_folders = [x.name for x in CONTEXT_DIR.iterdir() if x.is_dir()]
                folder_name = st.selectbox("Select the folder containing PDF files:", context_folders)
                bucket_name = None
            if source_type == "s3":
                bucket_name = st.text_input("Enter the S3 bucket name containing PDF files:")
                if not bucket_name:
                    st.warning("Invalid Bucket Name.")
                    st.stop()
                folder_name = None
                if bucket_name:
                    aws_config = get_aws_config()

            if st.button("Process Vector DB", key="process_btn"):
                with st.spinner("Processing..."):
                    documents = load_documents(source_type, folder_name, bucket_name)
                    if documents:
                        if action == "Create new vector database":
                            db_path = VECTOR_DB_DIR / db_name
                            if db_path.exists():
                                resync = st.checkbox(f"Vector database '{db_name}' already exists. Do you want to resync it?")
                                if resync:
                                    resync_vector_db(db_name, documents)
                            else:
                                create_vector_db(documents, db_name)
                        else:  # Resync existing vector database
                            resync_vector_db(db_name, documents)

        elif action == "Delete vector database":
            vector_databases = [x.name for x in VECTOR_DB_DIR.iterdir() if x.is_dir()]
            db_name = st.selectbox("Select the vector database to delete:", vector_databases)
            if st.button("Delete Vector DB", key="delete_btn"):
                with st.spinner("Deleting..."):
                    delete_vector_db(db_name)

    st.markdown("---")
    st.info("👆 Use the options above to manage your vector databases.")