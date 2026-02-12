'''
#Google Embeddings, Gemini Pro, BM25 RAG with Ensemble.
from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.document_loaders import UnstructuredPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import streamlit as st
import base64
import chromadb
'''

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Updated import
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
#from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import BM25Retriever
#from langchain_community.retrievers import EnsembleRetriever
#from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
#from langchain_core.retrievers import EnsembleRetriever
#from langchain.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import streamlit as st
import base64
import chromadb

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
groq_api_key = os.getenv("GROQ_API_KEY")

'''
def load_pdf_file(pdf): #pdf=bio.pdf
    data_file = UnstructuredPDFLoader(pdf)
    docs = data_file.load()
    return docs
'''

def load_pdf_file(pdf):
    data_file = PyPDFLoader(pdf)
    docs = data_file.load()
    return docs


def create_chunks_and_embeddings(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    #embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": 3})    
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k =  3    
    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,keyword_retriever],weights=[0.5, 0.5])
    return ensemble_retriever

def load_llm():
    #llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
    llm = ChatGroq(temperature=0.6, groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    return llm

def create_prompt_template(ensemble_retriever, llm):
    template = """
    <|system|>>
    You are a helpful AI Assistant that follows instructions extremely well.
    Use the following context to answer user question.
    The following context is all about a person named 'Pallavi'. Users should feel that they are talking with Pallavi not an AI bot.
    
    Think step by step before answering the question. You will get a $100 tip if you provide correct answer. 
    If the answer is not in provided context just say, "I am sorry, I don't have permission to provide you this information", don't provide the wrong answer.
    
    CONTEXT: {context}
    </s>
    <|user|>
    {query}
    </s>
    <|assistant|>
    """

    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    
    chain = (
        {"context": ensemble_retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )

    return chain

def get_base64_encoded_image(image_filename):
    with open(image_filename, 'rb') as f:
        encoded_string = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{encoded_string}"

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()










