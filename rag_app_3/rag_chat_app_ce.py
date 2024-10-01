import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from dotenv import load_dotenv
import os
from pathlib import Path

from rag_app_3.config import ModelProvider
from rag_app_3.models import setup_openai_model, setup_anthropic_model, setup_groq_model, setup_mistral_model, setup_embedding_model
from rag_app_3.prompts import ChatSystemPrompts as csp
from rag_app_3.vector_store import VectorStore
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class RAGChatAppCE:
    def __init__(self):
        self.initialize_session_state()
        self.vStore = VectorStore()

    def initialize_session_state(self):
        for key, default_value in [
            ("selected_model_ce", None),
            ("rag_ce_store", {}),
            ("rag_ce_messages", []),
            ("rag_ce_config", {"configurable": {"session_id": "rag_xyz123"}}),
            ("selected_vector_ce", None),
            ("use_vector_db_ce", False),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default_value

    def setup_llm(self, provider):
        model_setup = {
            ModelProvider.OPENAI.value: setup_openai_model,
            ModelProvider.ANTHROPIC.value: setup_anthropic_model,
            ModelProvider.GROQ.value: setup_groq_model,
            ModelProvider.MISTRAL.value: setup_mistral_model
        }
        return model_setup[provider]()

    def setup_sidebar(self):
        with st.sidebar:
            st.session_state.use_vector_db_ce = st.checkbox("Use Vector DB for Context", False)
            if st.session_state.use_vector_db_ce:
                available_vector_dbs = self.vStore.get_available_vector_dbs()
                if available_vector_dbs:
                    selected_vector = st.selectbox(
                        'Choose Vector DB:',
                        available_vector_dbs,
                        index=0,
                    )
                    # Update the selected vector store if it has changed
                    if selected_vector != st.session_state.selected_vector_ce:
                        st.session_state.selected_vector_ce = selected_vector
                        self.vStore.load_vector_store(selected_vector)
                        st.session_state.current_vector_db = selected_vector  # Track current vector DB
                        st.success(f"Vector DB updated to: {selected_vector}")  # Feedback to user
                        st.rerun()  # Rerun the script to reflect changes
                else:
                    st.warning('No Vector DB available. Please create a new one.')
            else:
                st.session_state.selected_vector_ce = None
                st.session_state.current_vector_db = None

        # Retrieve the current retriever based on user choice
        retriever = self.vStore.get_retriever() if st.session_state.use_vector_db_ce else None

        st.sidebar.header("Available LLM Model")
        selected_provider = st.sidebar.selectbox(
            "Choose Company:",
            [provider.value for provider in ModelProvider],
            index=0,
            key='llm_provider_ce'
        )

        return self.setup_llm(selected_provider), retriever

    def clear_chat(self):
        if st.sidebar.button('Clear Chat', key='clear_chat_ce'):
            st.session_state.rag_ce_messages = []
            st.session_state.rag_ce_store = {}
            st.rerun()

    @staticmethod
    def get_session_history(session_id: str):
        if session_id not in st.session_state.rag_ce_store:
            st.session_state.rag_ce_store[session_id] = ChatMessageHistory()
        return st.session_state.rag_ce_store[session_id]

    def setup_chain(self, llm, retriever):
        if st.session_state.use_vector_db_ce and retriever:
            # Use the existing RAG chain setup
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", csp.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", csp.computer_engineer_expert_system_prompt_rag),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            qa_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

            return RunnableWithMessageHistory(
                rag_chain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        else:
            # Use a simple chain without RAG
            qa_prompt = ChatPromptTemplate.from_messages([
            ("system", csp.computer_engineer_expert_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ])

        chain = qa_prompt | llm
        return RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            )

    def display_chat_messages(self):
        for message in st.session_state.rag_ce_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_user_input(self, conversational_chain):
        if prompt := st.chat_input("Hi! How can I help you?", key='chat_input_ce'):
            st.session_state.rag_ce_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    if st.session_state.use_vector_db_ce : 
                        with st.spinner("Thinking🤔"):
                            response = conversational_chain.invoke({"input": prompt}, config=st.session_state.rag_ce_config)
                            context_list = list(set(context.metadata['source'] for context in response['context']))
                            answer = f"{response['answer']}\n\n📌 Source: {', '.join(context_list)}"
                            response = st.write(answer)
                    else:
                        # When not using vector DB, response is an AIMessage
                        stream = conversational_chain.stream({"input": prompt}, config=st.session_state.rag_ce_config)
                        response = st.write_stream(stream)
                except Exception as e:
                    st.info(f"An error occurred: {str(e)}. Please check your API key or try again.")
                    st.stop()
            st.session_state.rag_ce_messages.append({"role": "assistant", "content": answer if st.session_state.use_vector_db_ce else response})


    def run(self):
        st.markdown(" #### Programming Language Expert 💻 ")
        llm, retriever = self.setup_sidebar()
        conversational_chain = self.setup_chain(llm, retriever)
        self.clear_chat()
        self.display_chat_messages()
        self.handle_user_input(conversational_chain)