# pip install streamlit langchain langchain-openai beautifulsoup4 langchain-community pyhon-dotenv chromadb pypdf

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import SitemapLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader

load_dotenv()

def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    # loader = SitemapLoader(web_path=url)
    # loader = RecursiveUrlLoader(url=url, extractor=lambda x: Soup(x, "html.parser").text)

    return loader

def get_vector_store_from_pdf(file):
    loader = PyPDFLoader(file)

    return loader

def get_vector_store_from_url_recursive(url):
    loader = RecursiveUrlLoader(url=url, extractor=lambda x: Soup(x, "html.parser").text)

    return loader

def merge_loader(url_loader, pdf_loader, url_recursive_loader):
    loader_all = MergedDataLoader(loaders=[url_loader, pdf_loader, url_recursive_loader])
    document = loader_all.load()

    # split document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create a vectorestore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store



def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retirever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query")
    ])

    retriever_chain = create_history_aware_retriever(llm, retirever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response['answer'] 


# app config
st.set_page_config(page_title="SATIA (Statistical AI Assistant)")
st.title("SATIA (Statistical AI Assistant)")

# sidebar
# with st.sidebar:
#     st.header("Settings")
#     website_url = st.text_input("Website URL")

# streamlit run src/app.py to run the GUI

# user input
url = "http://babel.bps.go.id/"
pdf_file = "./doc/layanan.pdf"
url_recursive = "https://babel.bps.go.id/pressrelease.html?katsubjek=&Brs%5Btgl_rilis_ind%5D=&Brs%5Btahun%5D=2024&yt0=Cari"

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Halo, saya adalah SATIA, Statistical AI Assistant. Apa yang bisa saya bantu?"),
    ]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = merge_loader(get_vectorstore_from_url(url), get_vector_store_from_pdf(pdf_file), get_vector_store_from_url_recursive(url_recursive))

user_query = st.chat_input("Tulis pesan anda di sini...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
