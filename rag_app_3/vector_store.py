import streamlit as st
from pathlib import Path
from shutil import rmtree
from langchain_community.document_loaders import S3DirectoryLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
import os
from rag_app_3.models import setup_embedding_model
import re
import time

class VectorStore:
    _widget_counter = 0

    def __init__(self):
        self.index_root_path = self.create_index_root_path()
        self.vector_store = None
        self.reset_session_state()

    def reset_session_state(self):
        for key in ['aws_region_name', 'aws_access_key', 'aws_secret_access_key', 'selected_vector', 'bucket_name', 'use_local_files', 'selected_folder']:
            if key not in st.session_state:
                st.session_state[key] = None
    
    @staticmethod
    def create_index_root_path():
        path = Path.cwd() / 'vector_store'
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def load_vector_store(self, index_root_path, _embedding_model, selected_vector):
        if selected_vector:
            return FAISS.load_local(
                str(index_root_path / selected_vector),
                embeddings=_embedding_model,
                allow_dangerous_deserialization=True
            )
        return None
    
    def load_index(self, embedding_model):
        available_vector_db = [file.name for file in self.index_root_path.glob('*') if file.is_dir()]
        if available_vector_db:
            unique_key = f'load_index_selectbox_{self._widget_counter}_{int(time.time() * 1000)}'
            self._widget_counter += 1
            st.session_state.selected_vector = st.selectbox('Choose Vector DB:', available_vector_db, key=unique_key)
            self.vector_store = self.load_vector_store(self.index_root_path, embedding_model, st.session_state.selected_vector)
        else:
            st.warning('No Vector DB available. Please create a new one.')

    def _create_index(self, embedding_model):
        try:
            if st.session_state.use_local_files:
                if not st.session_state.selected_folder:
                    st.error("No folder selected. Please select a folder containing PDF files.")
                    st.stop()
                index_path = self.index_root_path / f'{st.session_state.selected_folder}-index'
                st.toast(f"Creating index for local folder: {st.session_state.selected_folder}")
                context_folder = Path.home() / "context_folder" / st.session_state.selected_folder
                if not context_folder.exists():
                    st.error(f"The folder '{st.session_state.selected_folder}' does not exist in the context_folder.")
                    st.stop()
                loader = DirectoryLoader(str(context_folder), glob="**/*.pdf", loader_cls=PyPDFLoader)
                documents = loader.load()
            else:
                if not st.session_state.bucket_name:
                    st.error("Bucket name cannot be empty")
                    st.stop()
                aws_config = self._get_aws_config()
                bucket_name = st.session_state.bucket_name
                index_path = self.index_root_path / f'{bucket_name}-index'
                st.toast(f"Creating index for bucket: {bucket_name}")
                loader = S3DirectoryLoader(bucket_name, **aws_config)
                documents = loader.load()
            if len(documents) < 1:
                st.error(f"No documents found. Please check your {'selected folder' if st.session_state.use_local_files else 'S3 bucket'} and make sure it contains PDF files.")
                st.stop()
            
            st.toast(f"Loaded {len(documents)} documents")
            text_splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="gradient")
            docs = text_splitter.split_documents(documents=documents)
            st.toast(f"Split into {len(docs)} chunks")
            vectorstore_faiss = FAISS.from_documents(docs, embedding_model)
            vectorstore_faiss.save_local(str(index_path))
            st.toast(f"Index created and saved at {index_path}")
            return vectorstore_faiss
        except Exception as e:
            st.error(f"Error in _create_index: {str(e)}")
            raise

    def _get_aws_config(self):
        aws_config = {}
        for key, env_var, display_name in [
            ('region_name', 'AWS_REGION_NAME', 'AWS Region Name'),
            ('aws_access_key_id', 'AWS_ACCESS_KEY', 'AWS Access Key'),
            ('aws_secret_access_key', 'AWS_SECRET_ACCESS_KEY', 'AWS Secret Access Key')
        ]:
            value = os.getenv(env_var) or st.sidebar.text_input(
                f"Enter {display_name}:",
                type="password" if 'key' in key else "default",
                key=f"aws_config_{key}"
            )
            if not value:
                st.sidebar.warning(f"Invalid {display_name}")
                st.stop()
            aws_config[key] = value
        return aws_config

    def get_retriever(self):
        if self.vector_store:
            return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        return None

    def reset_index(self):
        available_vector_db = [file.name for file in self.index_root_path.glob('*') if file.is_dir()]
        if available_vector_db:
            selected_vector = st.session_state.selected_vector or st.selectbox('Choose Vector DB to reset:', available_vector_db, key='reset_index_selectbox')
            if not selected_vector:
                st.warning('No Vector DB selected.')
                st.stop()
            index_path = self.index_root_path / selected_vector
            rmtree(index_path)
            st.toast(f'{selected_vector} has been reset.')
            st.session_state.selected_vector = None
            self.vector_store = None
            st.rerun()
        else:
            st.warning('No Vector DB available.')
                
    @staticmethod
    def use_vector_db():
        return st.sidebar.toggle("Use Vector DB for Context", value=False, key='use_vector_db_toggle')
    
    def create_new_db(self, embedding_model):
        st.session_state.use_local_files = st.radio("Choose PDF source:", ("Local Files", "S3 Bucket"), key='pdf_source_radio_vs') == "Local Files"
        
        if st.session_state.use_local_files:
            context_folder = Path.home() / "context_folder"
            if not context_folder.exists():
                context_folder.mkdir(parents=True, exist_ok=True)
                st.info("The 'context_folder' has been created. Please add your PDF files to a subfolder within it.")
                st.stop()
            
            subfolders = [f.name for f in context_folder.iterdir() if f.is_dir()]
            if not subfolders:
                st.warning("No subfolders found in the context_folder. Please create a subfolder and add your PDF files to it.")
                st.stop()
            
            st.session_state.selected_folder = st.selectbox("Select a folder:", subfolders, key='folder_selectbox_vs')
        else:
            st.session_state.bucket_name = st.text_input("Enter S3 Bucket Name:", key='bucket_name_input_vs')
            if not st.session_state.bucket_name:
                st.warning("Bucket Name cannot be empty")
                st.stop()
            elif not re.match(r'^[a-zA-Z0-9.\-_]{1,255}$', st.session_state.bucket_name):
                st.warning("Invalid Bucket Name. It must contain only letters, numbers, hyphens, and underscores, and be between 1 and 255 characters long.")
                st.stop()

        if st.button('Create Vector DB', key='create_vector_db'):
            try:
                self.vector_store = self._create_index(embedding_model)
                st.toast("New Vector DB created successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error creating new Vector DB: {str(e)}")