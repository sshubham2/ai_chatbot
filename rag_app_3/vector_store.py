import streamlit as st
from pathlib import Path
from shutil import rmtree
from langchain_community.document_loaders import S3DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
import os
from rag_app_3.models import setup_embedding_model

class VectorStore:
    def __init__(self):
        self.index_root_path = self.create_index_root_path()
        self.vector_store = None
        if "aws_region_name" in st.session_state:
            st.session_state.aws_region_name = None
        if "aws_access_key" in st.session_state:
            st.session_state.aws_access_key = None
        if "aws_secret_access_key" in st.session_state:
            st.session_state.aws_secret_access_key = None
        if "selected_vector" in st.session_state:
            st.session_state.selected_vector = None
    
    def create_index_root_path(self):
        (Path.cwd()/'vector_store').mkdir(parents=True, exist_ok=True)
        return Path.cwd()/'vector_store'

    def load_or_create_index(self, embedding_model):
        with st.sidebar:
            available_vector_db = [file.name for file in self.index_root_path.glob('*')]
            if len(available_vector_db) > 0:
                st.session_state.selected_vector = st.selectbox('Choose Vector DB:', available_vector_db)
                self.vector_store = FAISS.load_local(str(self.index_root_path/st.session_state.selected_vector),
                                                        embeddings=embedding_model,
                                                        allow_dangerous_deserialization=True)
            else:
                st.toast('No Vector DB avilable.')
                self._create_index(setup_embedding_model())
                self.load_or_create_index(embedding_model)

    def _create_index(self, embedding_model):
        with st.sidebar:
            st.session_state.aws_region_name = os.getenv('AWS_REGION_NAME') or st.text_input("Enter AWS Region Name:")
            if not st.session_state.aws_region_name:
                st.warning("Invalid Region Name")
                st.stop()
            st.session_state.aws_access_key = os.getenv('AWS_ACCESS_KEY') or st.text_input("Enter AWS Access Key:", type="password")
            if not st.session_state.aws_access_key:
                st.warning("Invalid AWS Access Key")
                st.stop()
            st.session_state.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY') or st.text_input("Enter AWS Secret Access Key:", type="password")
            if not st.session_state.aws_secret_access_key:
                st.warning("Invalid AWS Secret Access Key")
                st.stop()
            bucket_name = st.text_input("Enter Bucket Name:")
            if not bucket_name:
                st.warning("Invalid Bucket Name")
                st.stop()
            index_path = self.index_root_path/f'{bucket_name}-index'
        loader = S3DirectoryLoader(bucket_name,
                                   region_name=st.session_state.aws_region_name,
                                   aws_access_key_id=st.session_state.aws_access_key,
                                   aws_secret_access_key=st.session_state.aws_secret_access_key)
        documents = loader.load()
        # text_splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="standard_deviation")
        text_splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="gradient")
        docs = text_splitter.split_documents(documents=documents)
        vectorstore_faiss = FAISS.from_documents(docs, embedding_model)
        vectorstore_faiss.save_local(str(index_path))

    def get_retriever(self):
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})

    def reset_index(self):
        with st.sidebar:
            available_vector_db = [file.name for file in self.index_root_path.glob('*')]
            if len(available_vector_db) > 0:
                selected_vector = st.session_state.selected_vector or st.selectbox('Choose Vector DB:', available_vector_db)
                if not selected_vector:
                    st.warning('No Vector DB selected.')
                    st.stop()
                index_path = self.index_root_path/selected_vector
                rmtree(index_path)
            else:
                st.warning('No Vector DB avilable.')