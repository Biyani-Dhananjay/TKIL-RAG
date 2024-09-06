import streamlit as st 
import os 
from PyPDF2 import PdfReader

st.set_page_config(page_title="TKIL MINING", page_icon=":shark:", layout="wide")


LOGO_URL_LARGE = r'Ellicium Transparent Background 1.png'
st.logo(LOGO_URL_LARGE,icon_image=LOGO_URL_LARGE)


path_to_store_data = "Input_data"
if not os.path.exists(path_to_store_data):
    os.makedirs(path_to_store_data)

uploaded_file  = st.sidebar.file_uploader("Upload a file", type=["pdf"] , key="file_uploader")
if uploaded_file:
    file_name = uploaded_file.name
    pdf_folder_path = os.path.join(path_to_store_data , file_name.split('.')[0])

    if not os.path.exists(pdf_folder_path):
        os.makedirs(pdf_folder_path)
    
    file_path = os.path.join(pdf_folder_path, file_name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    pdf_reader = PdfReader(file_path)
    number_of_pages = len(pdf_reader.pages)
    for page_num in range(number_of_pages):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        st.write(page_text)
        
names_of_folders = os.listdir(path_to_store_data)
selected_pdf_folder = st.sidebar.selectbox("Already Uploaded PDF's", names_of_folders,index=None, placeholder="Please select a pdf")

if selected_pdf_folder:
    st.write("PDF selected is ", selected_pdf_folder)
