import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from docx import Document
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
import json

# load_dotenv()
# google_api_key = os.getenv("GOOGLE_API_KEY")
config_file_path = 'config.json'

# Open and read the JSON file
with open(config_file_path, 'r') as file:
    config_data = json.load(file)

GOOGLE_API_Key = config_data['GOOGLE_API_Key']

# Configuring Google Generative AI
# genai.configure(api_key=google_api_key)

# Reading the text from the pdfs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

# Splitting the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    chunks = text_splitter.split_text(text)
    return chunks

# Convert the chunks into vectors
def get_vector_store(text_chunks):
    # embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
    embeddings = HuggingFaceEmbeddings()
    vector_store= FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('faiss_index')

# make a conversational chain
def get_conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answewr is not in provided context just say, "answer is not available in the context", don't provide the wrong answer \n\n
        Context: \n {context}?\n
        Question: \n{question}\n

        Answer: 
    """
    model = ChatGoogleGenerativeAI(model = 'gemini-pro', temperature = 0.7, google_api_key = GOOGLE_API_Key)
    prompt = PromptTemplate(template = prompt_template, input_variables = ['context', 'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt = prompt)

    return chain

# take the user input
def user_input(user_question):
    # embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
    embeddings = HuggingFaceEmbeddings()
    new_db = FAISS.load_local('faiss_index', embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, 'question': user_question}, return_only_outputs = True
    )
    # print(response)
    st.write("Reply: ", response['output_text'])

# Making an interface
def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("Chat With Multiple PDF using Gemini")

    user_question = st.text_input("Ask a question from the PDF Files")
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title('Menu:')
        pdf_docs = st.file_uploader("Upload you PDF Files and Click on Submit & Process Button", type='pdf',
            accept_multiple_files=True)

        if st.button('Submit & Process'):
            with st.spinner('Processing...'):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done and You can ask query.")

if __name__ == '__main__':
    main()
