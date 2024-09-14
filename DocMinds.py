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
      height: 2rem;
          width: 10rem;
    }
  </style>
        """)

st.header("Welcome to DocMinds by Ellicium")
st.write("You can upload your pdf documents here")
st.write("These documents will be converted into vectorstores")
st.write("You can chat with these pdf files in the **chat** section")
