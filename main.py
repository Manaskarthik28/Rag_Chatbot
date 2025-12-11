# Import all necessary libraries

from fastapi import FastAPI, Body
import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# Initialise the fast api app
app = FastAPI()

# load environment variables 'google key'
load_dotenv()

# create a rag function
def initialize_rag():
    # 1. Load Document
    file_path = 'data/NASA_blog.pdf'
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    texts = text_splitter.split_documents(documents)
    
    # 3. Create Embeddings and Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(texts, embeddings)
    
    # 4. Initialize the LLM (Gemini)
    llm = ChatGoogleGenerativeAI(
        model = 'gemini-2.5-flash',
        temperature = 0.7 
    )
    
    # 5. Create the RetrievalQA Chain
    rag_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff', 
        retriever = vectorstore.as_retriever() 
    )
    return rag_chain
# Initialize the RAG chain globally
rag_chain = initialize_rag()

# route user query
@app.post('/query')
async def user_query(request:str = Body(
   ...,  media_type = 'text/plain'
)):
    question = request
    rag_result = rag_chain.invoke(question)
    answer = rag_result['result']
    return {'question':question, 'answer':answer}

    


