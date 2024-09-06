# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 23:00:26 2024

@author: Yash_Ranjankar
"""

import streamlit as st
import save_graph_and_vectorstore

with st.sidebar:
    st.write("Upload PDFs here")
    uploaded_file = st.file_uploader("Choose a file")

st.header("Welcome to DocMinds by Ellicium")
st.write("You can upload your pdf documents here")
st.write("These documents will be converted into vectorstores")
st.write("You can chat with these pdf files in the **chat** section")

if uploaded_file is not None:
    text = save_graph_and_vectorstore.get_pdf_text(uploaded_file)
    st.write(text[:100])





