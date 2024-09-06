import streamlit as st
from glob import glob
import os
import pickle
import faiss
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
openai_api_key="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"
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# vectorstore folder
vectorstore_folder = r"D:\Ellicium Assignments\Projects\TKIL-RAG\vectorstore"
file_extention = ".pkl"
vectorstores = glob(os.path.join(vectorstore_folder,f"*{file_extention}"))
vectorstore_names = [os.path.basename(f).replace(f"{file_extention}","") for f in vectorstores]

def search_graph(graph, query_embedding, top_k=5):
    # Calculate similarity of query to all nodes in the graph
    node_embeddings = np.array([data['embedding'] for _, data in graph.nodes(data=True)])
    similarities = cosine_similarity([query_embedding], node_embeddings).flatten()
    print(similarities)
    # Get the top_k most similar nodes
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    relevant_nodes = [graph.nodes[i]['sentence'] for i in top_k_indices]
    
    return relevant_nodes

def load_graph_pkl(file_path):
    with open(file_path, 'rb') as f:
        G = pickle.load(f)
    return G

def load_faiss_index(file_path):
    # Load the FAISS index from disk
    index = faiss.read_index(file_path)
    return index

# Function to search the FAISS vector store and retrieve the most relevant chunks
def search_faiss(query_embedding, chunks, index, top_k=5):
    distances, indices = index.search(np.array([query_embedding]).astype(np.float32), k=top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def get_response_openai(System_Prompt: str, selected_model="gpt-4o"):
    """
    Function used for generating response form OpenAI model
    Here we are Passing the System Prompt and Extracted text from resume.
    """

    client = OpenAI(api_key=openai_api_key)
    
    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": System_Prompt}
                ],
            # response_format=response_format,
            temperature=0
            )
    except Exception as e:
        print(f"Error creating completion request for model '{selected_model}'")
        raise e

    return response.choices[0].message.content

def find_page_number(pdf_path, search_text):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if search_text in text:
                return page_num + 1
    return None

def gpt_prompt(user_query, context):
    prompt = f"""
Role: You are an expert in mechanical system design.

Task: Your objective is to respond to the given query using only the provided context, which will be extracted from a mechanical tender document.

Instructions:
1. Ensure your answer thoroughly addresses all necessary aspects required for designing the component specified in the query.
2. Do not include any extraneous information beyond what's needed for the design.
3. Categorize the response into the following sections in the order mentioned. if applicable to the component:
    Material and type
    Design Considerations
    Additional Reqirements
    Special Cases
    Summary 
4.Format your response strictly in .markdown. Do not use headers in markdown. Just add the bold formatting.

Context:
{context}

Query:
{user_query}
"""
    return prompt

st.set_page_config(page_title="DocMinds", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
LOGO_PATH = r'D:\Ellicium Assignments\Projects\TKIL-RAG\utility\Ellicium Transparent Background 1.png'
st.logo(LOGO_PATH,icon_image=LOGO_PATH)
# Create Radio buttons for all the availble vector stores
with st.sidebar:
    selected_option = st.radio("Choose a file",vectorstore_names)

st.header("Chat")
st.write(f"Selected option: **{selected_option}**")

# Load vector-store
graph = load_graph_pkl(os.path.join(vectorstore_folder, f"{selected_option}{file_extention}"))

## CHAT
# Initialize session state for storing conversation
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Function to display the chat messages
def display_chat():
    for message in st.session_state['messages']:
        if message['role'] == 'user':
            st.markdown(f":blue[**You: {message['content']}**]")
        else:
            st.markdown(f":red[**AI**]: {message['content']}")

# Text input for the user message
user_input = st.chat_input("Enter your query:")
# Submit user input to the model
if user_input:
    # Add the user message to session state
    st.session_state['messages'].append({"role": "user", "content": user_input})
    
    query_embedding = model.encode(user_input)
    
    retrieved_context = search_graph(graph, query_embedding, top_k=10)
    retrieved_context = '\n'.join(retrieved_context)
    prompt = gpt_prompt(user_input, retrieved_context)
    print(prompt)
    ai_message = get_response_openai(prompt, selected_model="gpt-4o")
    ai_message = ai_message.replace("markdown","")
    
    # ai_message = f"This was retireved_context: {retrieved_context}"
    st.session_state['messages'].append({"role": "assistant", "content": ai_message})
    
    # Display the updated chat
    display_chat()