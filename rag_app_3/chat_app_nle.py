# chat_app_ce.py
import streamlit as st
from typing import Dict, Any, List
import logging
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessageChunk, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
import PyPDF2
import docx2txt
import hashlib

from rag_app_3.config import ModelProvider
from rag_app_3.models import setup_openai_model, setup_anthropic_model, setup_groq_model, setup_mistral_model
from rag_app_3.prompts import ChatSystemPrompts as csp

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up caching
set_llm_cache(InMemoryCache())

class ChatAppNLE:
    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = None
        if "nle_store" not in st.session_state:
            st.session_state.nle_store = {}
        if "nle_messages" not in st.session_state:
            st.session_state.nle_messages = []
        if "nle_config" not in st.session_state:
            st.session_state.nle_config = {"configurable": {"session_id": "def123"}}
        if "use_file_upload" not in st.session_state:
            st.session_state.use_file_upload = False
        if "uploaded_file_content" not in st.session_state:
            st.session_state.uploaded_file_content = None
        if "uploaded_file_hash" not in st.session_state:
            st.session_state.uploaded_file_hash = None

    def setup_sidebar(self) -> Any:
        # Add toggle for file upload
        use_file_upload = st.sidebar.toggle("Enable File Upload", value=st.session_state.use_file_upload, key="file_upload_toggle")
        if use_file_upload != st.session_state.use_file_upload:
            st.session_state.use_file_upload = use_file_upload
            st.rerun()

        if st.session_state.use_file_upload:
            self.handle_file_upload()

        st.sidebar.header("Available LLM Model")
        selected_provider = st.sidebar.selectbox(
            "Choose Company:",
            [provider.value for provider in ModelProvider],
            index=0
        )

        if selected_provider == ModelProvider.OPENAI.value:
            return setup_openai_model()
        elif selected_provider == ModelProvider.ANTHROPIC.value:
            return setup_anthropic_model()
        elif selected_provider == ModelProvider.GROQ.value:
            return setup_groq_model()
        elif selected_provider == ModelProvider.MISTRAL.value:
            return setup_mistral_model()

    def handle_file_upload(self):
        uploaded_file = st.sidebar.file_uploader("Upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
        if uploaded_file is not None:
            file_content = uploaded_file.getvalue()
            file_hash = hashlib.md5(file_content).hexdigest()

            if file_hash != st.session_state.uploaded_file_hash:
                file_extension = uploaded_file.name.split(".")[-1].lower()
                if file_extension == "pdf":
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    content = ""
                    for page in pdf_reader.pages:
                        content += page.extract_text()
                elif file_extension == "docx":
                    content = docx2txt.process(uploaded_file)
                elif file_extension == "txt":
                    content = file_content.decode("utf-8")
                else:
                    st.sidebar.error("Unsupported file type")
                    return

                st.session_state.uploaded_file_content = content
                st.session_state.uploaded_file_hash = file_hash
                st.sidebar.success("File uploaded and processed successfully!")

    def clear_chat(self):
        with st.sidebar:
            if st.button('Clear Chat'):
                st.session_state.nle_messages = []
                st.session_state.nle_store = {}
                st.session_state.uploaded_file_content = None
                st.session_state.uploaded_file_hash = None
                st.rerun()

    @staticmethod
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.nle_store:
            st.session_state.nle_store[session_id] = ChatMessageHistory()
        return st.session_state.nle_store[session_id]

    def setup_chain(self, llm):
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", csp.natural_language_expert_sysmtem_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        chain = qa_prompt | llm
        return RunnableWithMessageHistory(chain, self.get_session_history, input_messages_key="messages")

    def display_chat_messages(self):
        for message in st.session_state.nle_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_user_input(self, conversational_rag_chain):
        # Add context from uploaded file if available
        if st.session_state.uploaded_file_content:
            context_message = f"Context from uploaded file:\n{st.session_state.uploaded_file_content}"
            conversational_rag_chain.invoke({"messages": context_message}, config=st.session_state.nle_config)

        prompt = st.chat_input(f"Hi! How can I help you?")
        if prompt:
            st.session_state.nle_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    stream = conversational_rag_chain.stream({"messages": prompt}, config=st.session_state.nle_config)
                    response = st.write_stream(stream)
                except Exception as e:
                    st.info(f"An error occurred: {str(e)}. Please check your API key or try again later.")
                    st.stop()

            st.session_state.nle_messages.append({"role": "assistant", "content": response})

    def run(self):
        st.markdown(" #### Natural Language Expert 💬 ")
        llm = self.setup_sidebar()
        conversational_rag_chain = self.setup_chain(llm)
        self.clear_chat()
        self.display_chat_messages()
        self.handle_user_input(conversational_rag_chain)