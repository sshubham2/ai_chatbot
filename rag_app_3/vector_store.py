import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import FAISS
from rag_app_3.models import setup_embedding_model

class VectorStore:
    def __init__(self):
        self.index_path = Path.home() / ".ragbot" / "vector_dbs"
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = None
        if "current_vector_db" not in st.session_state:
            st.session_state.current_vector_db = None

    def get_available_vector_dbs(self):
        return [file.name for file in self.index_path.glob('*') if file.is_dir()]

    def load_vector_store(self, selected_vector):
        if selected_vector and selected_vector != st.session_state.current_vector_db:
            try:
                st.session_state.vector_store = FAISS.load_local(
                    str(self.index_path / selected_vector),
                    embeddings=setup_embedding_model(),
                    allow_dangerous_deserialization=True
                )
                st.session_state.current_vector_db = selected_vector
                st.toast(f"{selected_vector} loaded successfully.")
            except Exception as e:
                st.error(f'Error loading vector store: {e}')
                st.session_state.vector_store = None
                st.session_state.current_vector_db = None
        return st.session_state.vector_store

    def get_retriever(self):
        if st.session_state.vector_store:
            return st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        return None