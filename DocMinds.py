import streamlit as st
import save_graph_and_vectorstore
import os
import json

with open("config.json", "r") as f:
    config = json.load(f)
    vector_store_folder = config["VECTOR_STORE_FOLDER"]
    pdf_path= config["PDF_PATH"]
    LOGO_PATH = config["LOGO_PATH"]

st.set_page_config(page_title="DocMinds", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
st.logo(LOGO_PATH,icon_image=LOGO_PATH)


st.html("""
  <style>
    [alt=Logo] {
      height: 45px;
          width: 8rem;
    }
  </style>
        """)

st.header("Welcome to DocMinds by Ellicium")
st.write("""Upload your PDF documents and let DocMinds convert them into smart, searchable vector stores. Use the power of AI to chat with your documents, ask questions, and extract key information in real-time, all in one platform.""")
st.markdown("""
- Upload PDF Files: Seamlessly integrate your files into DocMinds
- Smart Search: Access information instantly with AI-powered document search
- Interactive Chat: Chat with your documents to get insights or specific data points.
""")        
 
st.write("""Ready to dive into your documents?
Start by uploading your first PDF file or head to the chat section to explore your data.""")
st.write("You can chat with these pdf files in the **chat** section")
