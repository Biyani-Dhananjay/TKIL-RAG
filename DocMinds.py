import streamlit as st
import save_graph_and_vectorstore
import os
import json

with open("config.json", "r") as f:
    config = json.load(f)
    vector_store_folder = config["VECTOR_STORE_FOLDER"]
    pdf_path= config["PDF_PATH"]
    LOGO_PATH = config["LOGO_PATH"]

# Vector store folder
# vector_store_folder = r"D:\Ellicium Assignments\Projects\TKIL-RAG\vectorstore"

# LOGO_PATH = r'D:\Ellicium Assignments\Projects\TKIL-RAG\utility\Ellicium Transparent Background 1.png'
st.set_page_config(page_title="DocMinds", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
st.logo(LOGO_PATH,icon_image=LOGO_PATH)
# with st.sidebar:
#     st.write("Upload PDFs here")
#     uploaded_file = st.file_uploader("Choose a file")

st.header("Welcome to DocMinds by Ellicium")
st.write("You can upload your pdf documents here")
st.write("These documents will be converted into vectorstores")
st.write("You can chat with these pdf files in the **chat** section")

# if uploaded_file is not None:
#     # Get the text
#     text = save_graph_and_vectorstore.get_pdf_text(uploaded_file)
#     # Split and make text chunks
#     text_chunks = save_graph_and_vectorstore.chunk_text(text)
#     # Generate embeddings for the chunks
#     text_embeddings = save_graph_and_vectorstore.generate_sentence_embeddings(text_chunks)
#     # Build the graph using the embeddings
#     graph = save_graph_and_vectorstore.build_graph_from_embeddings(text_chunks, text_embeddings, similarity_threshold=0.5)
#     # Save graph
#     graph_filepath = os.path.join(vector_store_folder,uploaded_file.name.replace(".pdf",".graph.pkl"))
#     save_graph_and_vectorstore.save_graph_pickle(graph, graph_filepath)
#     # Write
#     st.markdown(f":blue[Created vector embeddings for **{uploaded_file.name}**]")