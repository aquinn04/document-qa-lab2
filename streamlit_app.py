import streamlit as st
from openai import OpenAI
import sys 

# Define each page, pointing to its actual file path
lab1_page = st.Page("pages/lab1.py", title="Lab 1", icon="📊")
lab2_page = st.Page("pages/lab2.py", title="Lab 2", icon="✏️")
lab3_page = st.Page("pages/lab3.py", title="Lab 3", icon="🔍")
lab4_page = st.Page("pages/lab4.py", title="Lab 4", icon="📈", default=True)
# Register them with navigation
pg = st.navigation([lab1_page, lab2_page, lab3_page, lab4_page])

st.set_page_config(page_title="Lab Apps", page_icon="🧑‍💻")

pg.run()