# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:33:00 2024

@author: Yash_Ranjankar
"""

import networkx as nx
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pdfplumber
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
import faiss
from langchain.vectorstores import FAISS
# Set your OpenAI API key
openai_api_key = "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"

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

def create_openai_embeddings(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    chunk_vectors = embeddings.embed_documents(chunks)
    return chunk_vectors


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

# Step 4: Perform Retrieval (e.g., using nearest neighbors in the graph)
def retrieve_relevant_nodes(graph, query_embedding, top_k=5):
    # Calculate similarity of query to all nodes in the graph
    node_embeddings = np.array([data['embedding'] for _, data in graph.nodes(data=True)])
    similarities = cosine_similarity([query_embedding], node_embeddings).flatten()
    
    # Get the top_k most similar nodes
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    relevant_nodes = [graph.nodes[i]['sentence'] for i in top_k_indices]
    
    return relevant_nodes

# Vector Storage: Store embeddings in a FAISS vector store
def store_vectors_in_faiss(chunk_vectors):
    dimension = len(chunk_vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_vectors).astype(np.float32))
    return index

# Function to search the FAISS vector store and retrieve the most relevant chunks
def search_faiss(query, chunks, index):
    embeddings = OpenAIEmbeddings()
    query_vector = embeddings.embed_query(query)
    distances, indices = index.search(np.array([query_vector]).astype(np.float32), k=5)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks



# Method 1
def get_response_openai(System_Prompt: str, selected_model="gpt-4"):
    """
    Function used for generating response form OpenAI model
    Here we are Passing the System Prompt and Extracted text from resume.
    """

    client = OpenAI(api_key=openai_api_key)
    # time.sleep(1)

    if selected_model in ['gpt-4-turbo-preview',
                          'gpt-3.5-turbo',
                          'gpt-4-0125-preview',
                          'gpt-4-1106-preview',
                          'gpt-3.5-turbo-0125',
                          'gpt-3.5-turbo-1106']:
        response_format = {"type": "json_object"}
    else:
        response_format = None
    
    response_format = None
    
    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": System_Prompt}
                ],
            response_format=response_format,
            temperature=0
            )
    except Exception as e:
        print(f"Error creating completion request for model '{selected_model}'")
        raise e

    return response.choices[0].message.content


# Sample Usage
if __name__ == "__main__":
    # Path to the PDF file
    pdf_path = r"C:/TKIL_Mining/Technical Specification.pdf"
    
    # Get text from the pdf
    entire_text = get_pdf_text(pdf_path)
    
    # Split and make text chunks
    text_chunks = chunk_text(entire_text)
    
    # Generate embeddings for the chunks
    # text_embeddings = generate_sentence_embeddings(text_chunks)
    text_embeddings = create_openai_embeddings(text_chunks)
    
    # Build the graph using the embeddings
    graph = build_graph_from_embeddings(text_chunks, text_embeddings, similarity_threshold=0.5)
    
    # Example query
    query = "I need to design a High Speed Coupling. Give me all the possible design specifications required which may include general instructions and technical specification."
    # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # query_embedding = model.encode(query)
    query_embedding = create_openai_embeddings(query)
    
    # Retrieve relevant nodes based on the query
    retrieved_info = retrieve_relevant_nodes(graph, query_embedding)
    
    print("Query:", query)
    print("Retrieved Information:", retrieved_info)
    
    System_Prompt = "\n".join(retrieved_info) + "\n\n" + query + "\n\nAnswer should be strictly from above mentioned context"
    response = get_response_openai(System_Prompt, selected_model="gpt-3.5-turbo")
    
    print(response)



