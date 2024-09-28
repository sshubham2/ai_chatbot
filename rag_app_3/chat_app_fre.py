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

load_dotenv()

class RAGChatAppFRE:
    def __init__(self):
        self.initialize_session_state()
        self.use_vector_db = VectorStore.use_vector_db()  # Call the static method
        if self.use_vector_db:
            self.vStore = VectorStore()
            self.embedding_model = setup_embedding_model()
        else:
            self.vStore = None
            self.embedding_model = None

    def initialize_session_state(self):
        for key, default_value in [
            ("selected_model_fre", None),
            ("rag_fre_store", {}),
            ("rag_fre_messages", []),
            ("rag_fre_config", {"configurable": {"session_id": "rag_ghi123"}}),
            ("use_local_files_fre", False),
            ("selected_folder_fre", None),
            ("create_new_db_fre", False)
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
            if self.use_vector_db:
                if st.button('Create New Vector DB'):
                    st.session_state.create_new_db_fre = True
                    st.rerun()

                if st.session_state.create_new_db_fre:
                    self.vStore.create_new_db_fre(self.embedding_model)
                    st.session_state.create_new_db_fre = False
                else:
                    self.vStore.load_index(self.embedding_model)
                
                col1, col2 = st.columns(2)
                if col1.button('Sync VectorDB', key='sync_vectordb_fre'):
                    self.vStore.reset_index()
                    self.vStore._create_index(self.embedding_model)
                if col2.button('Reset VectorDB', key='reset_vectordb_fre'):
                    self.vStore.reset_index()

        retriever = self.vStore.get_retriever() if self.use_vector_db else None

        st.sidebar.header("Available LLM Model")
        selected_provider = st.sidebar.selectbox(
            "Choose Company:",
            [provider.value for provider in ModelProvider],
            index=0,
            key='llm_provider_fre'
        )

        return self.setup_llm(selected_provider), retriever

    def clear_chat(self):
        if st.sidebar.button('Clear Chat', key='clear_chat_fre'):
            st.session_state.rag_fre_messages = []
            st.session_state.rag_fre_store = {}
            st.rerun()

    @staticmethod
    def get_session_history(session_id: str):
        if session_id not in st.session_state.rag_fre_store:
            st.session_state.rag_fre_store[session_id] = ChatMessageHistory()
        return st.session_state.rag_fre_store[session_id]

    def setup_chain(self, llm, retriever):
        if self.use_vector_db and retriever:
            # Use the existing RAG chain setup
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", csp.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", csp.financial_risk_expert_system_prompt_rag),
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
                ("system", csp.financial_risk_expert_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            chain = qa_prompt | llm

            return RunnableWithMessageHistory(
                chain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

    def display_chat_messages(self):
        for message in st.session_state.rag_fre_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_user_input(self, conversational_chain):
        if prompt := st.chat_input("Hi! How can I help you?", key='chat_input_fre'):
            st.session_state.rag_fre_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    if self.use_vector_db and self.vStore.vector_store:
                        with st.spinner("Thinking🤔"):
                            response = conversational_chain.invoke({"input": prompt}, config=st.session_state.rag_fre_config)
                            context_list = list(set(context.metadata['source'] for context in response['context']))
                            answer = f"{response['answer']}\n\n📌 Source: {', '.join(context_list)}"
                            response = st.write(answer)
                    else:
                        # When not using vector DB, response is an AIMessage
                        stream = conversational_chain.stream({"input": prompt}, config=st.session_state.rag_fre_config)
                        response = st.write_stream(stream)
                except Exception as e:
                    st.info(f"An error occurred: {str(e)}. Please check your API key or try again.")
                    st.stop()
            if self.use_vector_db and self.vStore.vector_store:
                st.session_state.rag_fre_messages.append({"role": "assistant", "content": answer})
            else:
                st.session_state.rag_fre_messages.append({"role": "assistant", "content": response})

    def run(self):
        st.markdown(" #### Financial Risk Expert 👓 ")
        llm, retriever = self.setup_sidebar()
        conversational_chain = self.setup_chain(llm, retriever)
        self.clear_chat()
        self.display_chat_messages()
        self.handle_user_input(conversational_chain)