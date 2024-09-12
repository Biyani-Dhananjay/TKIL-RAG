import streamlit as st
from glob import glob
import os
import pickle
import faiss
import json
from openai import OpenAI
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessed_prompt import high_speed_coupling_prompt,low_speed_coupling_prompt,gear_box_prompt

openai_api_key = st.secrets["OPENAI_API_KEY"]
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

with open("config.json", "r") as f:
    config = json.load(f)
    vectorstore_folder = config["VECTOR_STORE_FOLDER"]
    pdf_path= config["PDF_PATH"]
    LOGO_PATH = config["LOGO_PATH"]

file_extention = ".pkl"
vectorstores = glob(os.path.join(vectorstore_folder,f"*{file_extention}"))
vectorstore_names = [os.path.basename(f).replace(f"{file_extention}","") for f in vectorstores]

def search_graph(graph, query_embedding, top_k):
    # Calculate similarity of query to all nodes in the graph
    node_embeddings = np.array([data['embedding'] for _, data in graph.nodes(data=True)])
    similarities = cosine_similarity([query_embedding], node_embeddings).flatten()
    # Get the top_k most similar nodes
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    relevant_nodes = [graph.nodes[i]['sentence'] for i in top_k_indices]
    
    return relevant_nodes , top_k_indices

def load_graph_pkl(file_path):
    with open(file_path, 'rb') as f:
        G = pickle.load(f)
    return G

def load_faiss_index(file_path):
    # Load the FAISS index from disk
    index = faiss.read_index(file_path)
    return index

# Function to search the FAISS vector store and retrieve the most relevant chunks
def search_faiss(query_embedding, chunks, index, top_k):
    distances, indices = index.search(np.array([query_embedding]).astype(np.float32), k=top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def get_response_openai(System_Prompt: str):
    """
    Function used for generating response form OpenAI model
    Here we are Passing the System Prompt and Extracted text from resume.
    """
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {"role": "system", "content": System_Prompt}
                ],
            temperature=0
            )
        return response.choices[0].message.content

    except Exception as e:
        print("Error creating completion request for OpenAI")
        raise e

def get_adjacent_chunks(graph, index, k):
    """
    Retrieve the sentences for the node at index i and its k surrounding neighbors in the graph.
    """
    start_index = max(0, index - k)
    end_index = min(len(graph.nodes) - 1, index + k)
    # Retrieve sentences for the specified range of indices
    sentences = [graph.nodes[j]['sentence'] for j in range(start_index, end_index + 1)]
    context_string = "\n".join(sentences)
    return context_string    
    
def get_page_wise_text(pdf_path):
    entire_text = {}
    with pdfplumber.open(pdf_path) as pdf:
        # Iterate through all the pages
        for page_no , page  in enumerate(pdf.pages):
            entire_text[page_no + 1] = page.extract_text()
    return entire_text

def get_relevant_pages(retrieved_context,entire_text,indices,graph):
    related_context = {}
    related_context[0] = " "
    for i in range(len(retrieved_context)):
        found  = False
        for page_no, page_text in entire_text.items():
            if retrieved_context[i] in page_text:
                related_context[page_no] = page_text
                found = True
                break 
        
        if not found:
            for i in indices:
                context = get_adjacent_chunks(graph,i,k=2)
            # context = prev_context[i] + retrieved_context[i] + next_context[i]
                related_context[0] = related_context[0]+context + '\n'
            
    sorted_related_context = dict(sorted(related_context.items()))
    return sorted_related_context
                       
def gpt_prompt(user_query, context):
    prompt = f"""
Role: You are an expert in analyzing tenders and in mechanical system design.
 
Task: Your objective is to extract all relevant details for the specified component (e.g., gearboxes, high-speed couplings, shafts) from the tender document, focusing on specifications needed for pre-bid quotations to be shared with vendors.
 
Instructions:
1. Extract Component-Specific Specifications:
    1.1 It is extremely important to get every specification related to the query.
    1.2 Include detailed technical specifications related to the component mentioned in the query (e.g high-speed couplings, gearboxes), such as material requirements, design preferences, performance criteria, service factors, and relevant standards.
    1.3 Include specifications for associated subcomponents (e.g., shafting specifications, deflection in line shaft, reducers, holdback devices) when querying about gearboxes or similar components
    1.4 Exclude details related to other unrelated components (e.g., if the query is about high-speed couplings, exclude information about low-speed couplings).
 
2. Include Relevant General Requirements:
    2.1 Extract and include common points from the design basis that are critical for vendors, such as noise specifications, operational requirements, and general design standards.
    2.2 Ensure that these general points are highlighted if they apply to the component being queried.
    2.3 Include specific operational requirements, such as service factors (e.g., reducer service factors, torque ratios) when applicable to the component being queried.
 
3. Exclude Unnecessary Information:
    3.1 Avoid unrelated details such as information about other components not mentioned in the query (e.g., if the query is about high-speed couplings, exclude information about low-speed couplings).
    3.2 Exclude installation guides, safety measures, and operational details unless they directly impact the design or specification of the queried component.
 
4. Focus on Vendor-Ready Specifications:
    4.1 Ensure the extracted information is ready to be shared with a vendor for pre-bid quotations. Focus on the most critical specifications that a vendor would need for providing an accurate quotation.
 
5. Use Exact Wording:
    5.1 Use the exact wordings available in the provided context to ensure precision.
 
6. Formatting:
    6.1 Format your response strictly in markdown.
    6.2 Do not use headers in markdown.
    6.3 Use **bold formatting** to emphasize key points.
 
Context:
{context}
 
Important Design Consideration for all the components:
The plant & equipment shall be designed and sized based on the following
basic parameters including physical characteristics of raw materials and
products to be handled.
Noise level : 110 dB at a distance of 1m from the source of
noise and at a height of 1.2 m above floor level
mainly from cone crusher. For other equipment
noise level shall be 85 dB.
 
Query:
{user_query}
Add any noise-related considerations if available
"""
    return prompt


st.set_page_config(page_title="DocMinds", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
st.logo(LOGO_PATH,icon_image=LOGO_PATH)

with st.sidebar:
    selected_option = st.radio("Choose a file",vectorstore_names)

st.header("Chat")
st.write(f"Selected option: **{selected_option}**")

# Load vector-store
graph = load_graph_pkl(os.path.join(vectorstore_folder, f"{selected_option}{file_extention}"))



if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Function to display the chat messages
def display_chat():
    for message in st.session_state['messages']:
        if message['role'] == 'user':
            st.markdown(f":blue[You: {message['content']}]")
        else:
            st.markdown(f":red[AI]: {message['content']}")

if st.session_state['messages']:
    display_chat()
user_input = st.chat_input("What do you need:")

if user_input:
    st.session_state['messages'].append({"role": "user", "content": user_input})
    if 'low speed couplings' in user_input.lower().replace("-"," ") or 'low speed coupling' in user_input.lower().replace("-"," "):
        # query = "I need to design low speed coupling.Also give requirements for designing couplings"
        # print(query)
        prompt,pages = low_speed_coupling_prompt()
        ai_message = get_response_openai(prompt)
        ai_message += f"\n\nPages: {pages}"
        st.session_state['messages'].append({"role": "assistant", "content": ai_message})
        # display_chat()
    elif 'high speed couplings' in user_input.lower().replace("-"," ") or 'high speed coupling' in user_input.lower().replace("-"," "):
        prompt,pages = high_speed_coupling_prompt()
        ai_message = get_response_openai(prompt)
        ai_message += f"\n\nPages: {pages}"
        st.session_state['messages'].append({"role": "assistant", "content": ai_message})
    elif 'gear box' in user_input.lower().replace("-"," ") or 'gearbox' in user_input.lower().replace("-"," "):
        prompt,pages = gear_box_prompt()
        ai_message = get_response_openai(prompt)
        ai_message += f"\n\nPages: {pages}"
        st.session_state['messages'].append({"role": "assistant", "content": ai_message})
    else:
        query = user_input
    
        query_embedding = model.encode(query)
        
        retrieved_context,indices = search_graph(graph, query_embedding, top_k=10)
        
        # for chunk in retrieved_context:
        #     print(chunk)
        #     print("*"*100)

        # print("-"*100)
        entire_text = get_page_wise_text(pdf_path)
        relevant_pages = get_relevant_pages(retrieved_context, entire_text , indices , graph)
        retrieved_context_whole = list(relevant_pages.values())
        retrieved_pages = list(relevant_pages.keys())
        pages = ''
        for i in range(len(retrieved_pages)-1):
            pages+=f"{retrieved_pages[i+1]},"
        pages = pages[:-1]
        # for chunk in retrieved_context_whole:
        #     print(chunk)
        #     print("*"*100)
        
        prompt = gpt_prompt(query, retrieved_context_whole)
        ai_message = get_response_openai(prompt)
        ai_message = ai_message.replace("markdown","")
        ai_message += f"\n\nPages: {pages}"
        st.session_state['messages'].append({"role": "assistant", "content": ai_message})
        
    if st.session_state['messages']:
        latest_response = st.session_state['messages'][-1]['content']
        if latest_response:
            st.sidebar.download_button(
                data = latest_response.replace("**", ""),
                label = "Download Response",
                file_name = "Tendor-Specifications",
                mime = "text/plain"
            )
    display_chat() 