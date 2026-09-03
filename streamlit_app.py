import streamlit as st
from openai import OpenAI

# Define each page, pointing to its actual file path
lab1_page = st.Page("pages/lab1.py", title="Lab 1", icon="📊")
lab2_page = st.Page("pages/lab2.py", title="Lab 2", icon="✏️", default=True)

# Register them with navigation
pg = st.navigation([lab1_page, lab2_page])

st.set_page_config(page_title="Lab Apps", page_icon="🧑‍💻")

pg.run()