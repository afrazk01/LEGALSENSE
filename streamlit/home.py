import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import pandas as pd
import numpy as np
from scripts.qa_engine import get_legal_answer

st.title("Intelligent Legal Advisor ⚖️")
st.write("Welcome to the Intelligent Legal Advisor!")
st.markdown("Ask your legal questions based on the Pakistan Penal Code.")

query = st.text_input("Enter your legal question")

if query:
    with st.spinner("Searching legal documents..."):
        answer = get_legal_answer(query)
        st.success(f"**Answer:** {answer}")