# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:04:59 2024

@author: Yash_Ranjankar
"""
import networkx as nx
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pdfplumber
# from langchain.embeddings import OpenAIEmbeddings
# from openai import OpenAI
import pickle
import os
import faiss
# from langchain.vectorstores import FAISS

def get_pdf_text(pdf_path):
    # Variable to hold the entire text
    entire_text = ""
    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        # Iterate through all the pages
        for page in pdf.pages:
            # Extract text from the page and concatenate it
            text = page.extract_text()
            entire_text += text + "\n"  # Add a newline to separate pages
    return entire_text

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Set chunk size as per your requirement
        chunk_overlap=50  # Overlap between chunks to maintain context
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Step 2: Generate semantic vectors for the sentences
def generate_sentence_embeddings(sentences):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Pre-trained sentence embedding model
    embeddings = model.encode(sentences)
    return embeddings

# Step 3: Build the graph using semantic vectors
def build_graph_from_embeddings(sentences, embeddings, similarity_threshold=0.5):
    G = nx.Graph()
    
    # Add nodes (sentence embeddings)
    for i, sentence in enumerate(sentences):
        G.add_node(i, sentence=sentence, embedding=embeddings[i])
    
    # Add edges based on cosine similarity
    similarities = cosine_similarity(embeddings)
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarities[i][j] >= similarity_threshold:
                G.add_edge(i, j, weight=similarities[i][j])
    return G

# Vector Storage: Store embeddings in a FAISS vector store
def store_vectors_in_faiss(chunk_vectors):
    dimension = len(chunk_vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_vectors).astype(np.float32))
    return index

# Function to search the FAISS vector store and retrieve the most relevant chunks
def search_faiss(query_embedding, chunks, index, top_k=5):
    distances, indices = index.search(np.array([query_embedding]).astype(np.float32), k=top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def search_graph(graph, query_embedding, top_k=5):
    # Calculate similarity of query to all nodes in the graph
    node_embeddings = np.array([data['embedding'] for _, data in graph.nodes(data=True)])
    similarities = cosine_similarity([query_embedding], node_embeddings).flatten()
    
    # Get the top_k most similar nodes
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    relevant_nodes = [graph.nodes[i]['sentence'] for i in top_k_indices]
    
    return relevant_nodes

def save_graph_pickle(graph, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Graph saved to {file_path}")
        
def save_faiss_index(faiss_index, file_path):
    # Save the FAISS index to disk
    faiss.write_index(faiss_index, file_path)
    print(f"FAISS index saved to {file_path}")

def load_faiss_index(file_path):
    # Load the FAISS index from disk
    index = faiss.read_index(file_path)
    return index

def load_graph_pkl(file_path):
    with open(file_path, 'rb') as f:
        G = pickle.load(f)
    return G

# Sample Usage
if __name__ == "__main__":
    # Vector store folder
    vector_store_folder = r"C:\TKIL_Mining\vectorstore"
    
    # Path to the PDF file
    pdf_path = r"C:/TKIL_Mining/Technical Specification.pdf"
    
    # Get text from the pdf
    entire_text = get_pdf_text(pdf_path)
    
    # Split and make text chunks
    text_chunks = chunk_text(entire_text)
    
    # Generate embeddings for the chunks
    text_embeddings = generate_sentence_embeddings(text_chunks)
    # text_embeddings = create_openai_embeddings(text_chunks)
    
    # Build the graph using the embeddings
    graph = build_graph_from_embeddings(text_chunks, text_embeddings, similarity_threshold=0.5)
    # Save graph
    graph_filepath = os.path.join(vector_store_folder,os.path.basename(pdf_path).replace(".pdf",".graph.pkl"))
    save_graph_pickle(graph, graph_filepath)
    
    # Store vectors in FAISS
    faiss_index = store_vectors_in_faiss(text_embeddings)
    # Save FAISS
    faiss_filepath = os.path.join(vector_store_folder,os.path.basename(pdf_path).replace(".pdf",".index"))
    save_faiss_index(faiss_index, faiss_filepath)
    
    query = "I need to design a High Speed Coupling. Give me all the possible design specifications required which may include general instructions and technical specification."
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    
    # Search Faiss
    faiss_index = load_faiss_index(faiss_filepath)
    relevant_chunks_faiss = search_faiss(query_embedding, text_chunks, faiss_index, top_k=10)
    print(relevant_chunks_faiss)
    
    
    # Temp
    
    # Search Graph
    graph = load_graph_pkl(graph_filepath)
    relevant_chunks_graph = search_graph(graph, query_embedding, top_k=10)
    print(relevant_chunks_graph)
    
    a = set(relevant_chunks_faiss)
    b = set(relevant_chunks_graph)
    
    len(a)
    len(b)    
    len(a&b)
    len(a-b)
    len(b-a)
    










