import streamlit as st

home_page = st.Page("home.py", title="Home", icon="🏚️")
programming_language_expert = st.Page("computer_expert.py", title="Programming Language Expert", icon="💻")
natural_languag_expert = st.Page("language_expert.py", title="Natural Language Expert", icon="💬")
financial_risk_expert = st.Page("finance_risk_expert.py", title="Financial Risk Expert", icon="👓")
legal_expert = st.Page("legal_expert.py", title="Legal Expert", icon="🏛️")
vec_db_mgnmnt = st.Page("vec_db_mng.py", title="Vector Database Management", icon="🛠️")
# rag_ple = st.Page("rag_chatbot.py", title="Programming Language Expert (RAG)", icon="💻")

pg = st.navigation([home_page, programming_language_expert, natural_languag_expert, financial_risk_expert, legal_expert, vec_db_mgnmnt])
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout='wide')

pg.run()