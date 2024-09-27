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
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate

from rag_app_3.config import ModelProvider
from rag_app_3.models import setup_openai_model, setup_anthropic_model, setup_groq_model, setup_mistral_model, setup_embedding_model
from rag_app_3.prompts import ChatSystemPrompts as csp

from rag_app_3.vector_store import VectorStore

# Load environment variables
load_dotenv()

class RAGChatAppCE:
    def __init__(self):
        self.initialize_session_state()
        self.vStore = VectorStore()
    
    def initialize_session_state(self):
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = None
        if "rag_ce_store" not in st.session_state:
            st.session_state.rag_ce_store = {}
        if "rag_ce_messages" not in st.session_state:
            st.session_state.rag_ce_messages = []
        if "rag_ce_config" not in st.session_state:
            st.session_state.rag_ce_config = {"configurable": {"session_id": "rag_abc123"}}
        
    
    def setup_sidebar(self) -> Any: 
        self.vStore.load_or_create_index(setup_embedding_model())
        with st.sidebar:
            col1, col2 = st.columns(2)
            with col1:
                if st.button('Sync VectorDB'):
                    self.vStore.reset_index()
                    self.vStore._create_index(setup_embedding_model())
            with col2:
                if st.button('Reset VectorDB'):
                    self.vStore.reset_index()
        
        retriever = self.vStore.get_retriever()
        
        st.sidebar.header("Available LLM Model")
        selected_provider = st.sidebar.selectbox(
            "Choose Company:",
            [provider.value for provider in ModelProvider],
            index=0
        )
        
        if selected_provider == ModelProvider.OPENAI.value:
            return setup_openai_model(), retriever
        elif selected_provider == ModelProvider.ANTHROPIC.value:
            return setup_anthropic_model(), retriever
        elif selected_provider == ModelProvider.GROQ.value:
            return setup_groq_model(), retriever
        elif selected_provider == ModelProvider.MISTRAL.value:
            return setup_mistral_model(), retriever
        
    def clear_chat(self):
        with st.sidebar:
            if st.button('Clear Chat'):
                st.session_state.rag_ce_messages = []
                st.session_state.rag_ce_store = {}
                st.rerun()
                
    @staticmethod
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.rag_ce_store:
            st.session_state.rag_ce_store[session_id] = ChatMessageHistory()
        return st.session_state.rag_ce_store[session_id]
    
    def setup_chain(self, llm, retriever):
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", csp.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", csp.computer_engineer_expert_system_prompt_rag),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        return conversational_rag_chain
    
    def display_chat_messages(self):
        for message in st.session_state.rag_ce_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def handle_user_input(self, conversational_rag_chain):
        if prompt := st.chat_input(f"Hi! How can I help you?"):
            st.session_state.rag_ce_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking🤔"):
                    response = conversational_rag_chain.invoke({"input":prompt}, config=st.session_state.rag_ce_config)
                context_list = []
                for context in response['context']:
                    if context.metadata['source'] not in context_list:
                        context_list.append(context.metadata['source'])
                answer = response['answer'] + "\n\n📌 Source: " + (", ".join(context_list))
                try:
                    response = st.write(answer)
                except:
                    st.info("Invalid API Key? Please double check your API key.")
                    st.stop()
            st.session_state.rag_ce_messages.append({"role": "assistant", "content": answer})
            
    def run(self):
        st.markdown(" #### Programming Language Expert 💻 ")
        llm, retriever = self.setup_sidebar()
        conversational_rag_chain = self.setup_chain(llm, retriever)
        self.clear_chat()
        self.display_chat_messages()
        self.handle_user_input(conversational_rag_chain)