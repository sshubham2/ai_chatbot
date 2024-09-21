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

from rag_app_3.config import ModelProvider
from rag_app_3.models import setup_openai_model, setup_anthropic_model, setup_groq_model, setup_mistral_model
from rag_app_3.prompts import ChatSystemPrompts as csp

# Load environment variables
load_dotenv()

class ChatAppFRE:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = None
        if "fre_store" not in st.session_state:
            st.session_state.fre_store = {}
        if "fre_messages" not in st.session_state:
            st.session_state.fre_messages = []
        if "fre_config" not in st.session_state:
            st.session_state.fre_config = {"configurable": {"session_id": "def123"}}
        
    
    def setup_sidebar(self) -> Any:
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
        
    def clear_chat(self):
        with st.sidebar:
            if st.button('Clear Chat'):
                st.session_state.fre_messages = []
                st.session_state.fre_store = {}
                st.rerun()
                
    @staticmethod
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.fre_store:
            st.session_state.fre_store[session_id] = ChatMessageHistory()
        return st.session_state.fre_store[session_id]
    
    def setup_chain(self, llm):
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", csp.financial_risk_expert_system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        chain = qa_prompt | llm
        return RunnableWithMessageHistory(chain, self.get_session_history, input_messages_key="messages")
    
    def display_chat_messages(self):
        for message in st.session_state.fre_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def handle_user_input(self, conversational_rag_chain):
        prompt = st.chat_input(f"Hi! How can I help you?")
        if prompt:
            st.session_state.fre_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    stream = conversational_rag_chain.stream({"messages": prompt}, config=st.session_state.fre_config)
                    response = st.write_stream(stream)
                except Exception as e:
                    st.info(f"An error occurred: {str(e)}. Please check your API key or try again later.")
                    st.stop()

            st.session_state.fre_messages.append({"role": "assistant", "content": response})
            
    def run(self):
        st.markdown(" #### Financial Risk Expert 👓 ")
        llm = self.setup_sidebar()
        conversational_rag_chain = self.setup_chain(llm)
        self.clear_chat()
        self.display_chat_messages()
        self.handle_user_input(conversational_rag_chain)
                
    
        
    
    
    