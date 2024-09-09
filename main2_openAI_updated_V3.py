# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:09:31 2024

@author: Shubham_Patidar
"""

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import  HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import glob
import os
import pickle
import openai
import requests
from langchain.retrievers import ParentDocumentRetriever
from flask import Flask,request
from langchain.llms import OpenAI
from flask import Flask, request, Response, stream_with_context, json
import requests
import sseclient
import json
import streamlit as st

OPEN_AI_KEY = st.secrets["OPEN_AI_KEY"]

DEVICE = "cuda:6" if torch.cuda.is_available() else "cpu"

# vectorstore.save_local("faiss_index")
bi_enc_dict = {'mpnet-base-v2':"all-mpnet-base-v2",
              'instructor-large': 'hkunlp/instructor-large',
              'FlagEmbedding': 'BAAI/bge-large-en-v1.5'}
embeddings = HuggingFaceInstructEmbeddings(model_name=bi_enc_dict['FlagEmbedding'],
                                           query_instruction='Represent the question for retrieving supporting paragraphs: ',
                                           embed_instruction='Represent the paragraph for retrieval: ', model_kwargs={"device": DEVICE})

# vectorstore = FAISS.load_local("D:/KBR-FAA/kbr_llm/test_gpt_spine/spine_vectorstore", embeddings)
# OPEN_AI_KEY = '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'
# api_key = "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"
# Set the API key as an environment variable
# os.environ["OPENAI_API_KEY"] = api_key
# Optionally, check that the environment variable was set correctly
print("OPENAI_API_KEY has been set!")


#model = "gpt-3.5-turbo"
model = "gpt-4-0125-preview"

client = openai.OpenAI()

loaded_vectorstores = {}

def load_vectorstore(vectorstorename):
    global loaded_vectorstores
    if vectorstorename in loaded_vectorstores:
        # If vectorstore already loaded, return the loaded object
        return loaded_vectorstores[vectorstorename]

    # If vectorstore not loaded, load it
    root_path = f"/home/ubuntu/pc_vectorstores/{vectorstorename}/"
    if os.path.exists(root_path):
        with open(root_path+'parent_store_200_50.pickle', 'rb') as f:
            faa_store = pickle.load(f)
 
        faa_vectorstore = FAISS.load_local(root_path+"child_store_200_50", embeddings)

        child_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)

        faa_retriever = ParentDocumentRetriever(
            vectorstore=faa_vectorstore,
            docstore=faa_store,
            child_splitter=child_splitter,
            use_gpu = True,
            search_kwargs={"k": 10},
        )
            
        loaded_vectorstores[vectorstorename] = faa_retriever
        return faa_retriever
    else:
        print(f"Vectorstore '{vectorstorename}' not found.")
        return None
    
def rag(retriever, topic):
    docs = retriever.get_relevant_documents(topic)
    references = [i.metadata for i in docs]
    doc = [f'Context {i+1} : \n\t{docs[i].page_content}' for i in range(len(docs))]
    context = '\n\n'.join(doc)

    sys_prompt =  f"[INST]<>Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.<>[/INST]"

    template = "Context:\n"+context+"\n question: "+ topic
    all_references ='\n\n'+'References : '+''
    url = 'https://documents-evidence-report.s3.us-east-1.amazonaws.com/'
    m = 1
    for i in references:
        pdfname = i['pdf_name '].replace(' ', '%20')
        all_references = all_references+'\n'+str(m)+' - '+url+ pdfname  + '#page='+ i['page_number ']
        m+=1
    
    return docs, template, references

#***************************************************************************************************
chat_history = {}
def ret_hist(vectorstorename):
    try:
        return ''.join(chat_history[vectorstorename])
    except:
        return ''
prev = ['abc']        
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/HR_Navigator', methods=['POST'])
def generate_evdnc_report():
    abc = request.get_json()
    question = abc.get('question')
    vectorstorename = abc.get('usecase')
    if(prev[0] != vectorstorename):
        chat_history[vectorstorename] = []
        prev[0] = vectorstorename
        
    
    faa_retriever =load_vectorstore(vectorstorename)
    docs, prompt, references = rag(faa_retriever, question)
    
    def generate():
            hist = ''
            hist = hist+'Question: '+ question
            url = 'https://api.openai.com/v1/chat/completions'
            headers = {
                'content-type': 'application/json; charset=utf-8',
                'Authorization': 'Bearer '            
            }
            sys_prompt =  f"[INST]<>Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer, you can do some formal greetings<>[/INST]"


            payload = {
                'model': 'gpt-4-0125-preview',
                'messages': [
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content':  prompt}
                ],
                'temperature': 0.001, 
                'max_tokens': 2000,
                'stream': True,            
            }
            answer = ''
            response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)
            client = sseclient.SSEClient(response)
            for event in client.events():
                if event.data != '[DONE]':
                    try:
                        text = json.loads(event.data)['choices'][0]['delta']['content']
                        answer = answer+text
                        yield(text)
                    except:
                        yield('')
                   
            hist = hist+'Answer: '+answer
            if chat_history.get(vectorstorename):
                chat_history[vectorstorename] = chat_history[vectorstorename].append(hist)
            else:
                chat_history[vectorstorename] = []
                chat_history[vectorstorename].append(hist)
            db = FAISS.from_documents(docs,embeddings)
            score = db.similarity_search_with_score(answer,k=5,fetch_k = 5)
            refrence = '\n\n'+'References: \n'
            URL = 'https://elli-chatbot.s3.amazonaws.com/documents-internal-demo/'
            for i in range (len(score)):
                x = URL+score[i][0].metadata['pdf_name ']+'#page='+score[i][0].metadata['page_number ']
                x = x.replace(" ","%20")
                refrence = refrence+'\n'+x
            yield refrence
    return Response(stream_with_context(generate()))




if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=4444)    
    
