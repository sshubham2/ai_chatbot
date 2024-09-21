# models.py
import os
from typing import Any
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from rag_app_3.config import OPENAI_MODELS, ANTHROPIC_MODELS, GROQ_MODELS, MISTRAL_MODELS

def setup_openai_model() -> ChatOpenAI:
    """Set up and return an OpenAI model."""
    model_options = list(OPENAI_MODELS.keys())
    selected_model = st.sidebar.selectbox("Choose model:", model_options, index=0)
    st.session_state.selected_model = OPENAI_MODELS[selected_model]
    with st.sidebar:
        api_key = os.getenv('OPENAI_API_KEY') or st.text_input("Enter OpenAI API Key", type="password")
        if not api_key:
            st.warning("Invalid API Key")
            st.stop()

    try:
        return ChatOpenAI(
            model=st.session_state.selected_model,
            temperature=0.7,
            max_tokens=4096,
            timeout=None,
            max_retries=2,
            api_key=api_key
        )
    except Exception as e:
        st.error(f"Error setting up OpenAI model: {str(e)}")
        st.stop()

def setup_anthropic_model() -> ChatAnthropic:
    """Set up and return an Anthropic model."""
    model_options = list(ANTHROPIC_MODELS.keys())
    selected_model = st.sidebar.selectbox("Choose model:", model_options, index=0)
    st.session_state.selected_model = ANTHROPIC_MODELS[selected_model]
    with st.sidebar:
        api_key = os.getenv('ANTHROPIC_API_KEY') or st.text_input("Enter Anthropic API Key", type="password")
        if not api_key:
            st.warning("Invalid API Key")
            st.stop()

    try:
        return ChatAnthropic(
            model=st.session_state.selected_model,
            temperature=0.7,
            max_tokens=4096,
            timeout=None,
            max_retries=2,
            top_p=0.9,
            top_k=40,
            api_key=api_key
        )
    except Exception as e:
        st.error(f"Error setting up Anthropic model: {str(e)}")
        st.stop()
        
def setup_groq_model() -> ChatGroq:
    """Set up and return an Groq model."""
    model_options = list(GROQ_MODELS.keys())
    selected_model = st.sidebar.selectbox("Choose model:", model_options, index=0)
    st.session_state.selected_model = GROQ_MODELS[selected_model]
    with st.sidebar:
        api_key = os.getenv('GROQ_API_KEY') or st.text_input("Enter Groq API Key", type="password")
        if not api_key:
            st.warning("Invalid API Key")
            st.stop()

    try:
        return ChatGroq(
            model=st.session_state.selected_model,
            temperature=0.7,
            max_tokens=4096,
            timeout=None,
            max_retries=2,
            api_key=api_key
        )
    except Exception as e:
        st.error(f"Error setting up Groq model: {str(e)}")
        st.stop()
        
def setup_mistral_model() -> ChatMistralAI:
    """Set up and return an Mistral model."""
    model_options = list(MISTRAL_MODELS.keys())
    selected_model = st.sidebar.selectbox("Choose model:", model_options, index=0)
    st.session_state.selected_model = MISTRAL_MODELS[selected_model]
    with st.sidebar:
        api_key = os.getenv('MISTRAL_API_KEY') or st.text_input("Enter Mistral API Key", type="password")
        if not api_key:
            st.warning("Invalid API Key")
            st.stop()

    try:
        return ChatMistralAI(
            model=st.session_state.selected_model,
            temperature=0.7,
            max_tokens=4096,
            max_retries=2,
            top_p=0.9,
            top_k=40,
            api_key=api_key
        )
    except Exception as e:
        st.error(f"Error setting up Mistral model: {str(e)}")
        st.stop()