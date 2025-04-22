import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Environment Variables
os.environ['LANGCHAIN_API_KEY'] ="" #Use your Langchain api key"""
os.environ['LANGCHAIN_TRACKING_V2'] = 'True'
os.environ['LANGCHAIN_PROJECT'] = 'default'

# Initialize components
embedding_model = OllamaEmbeddings(model="gemma:2b")
llm = Ollama(model="gemma:2b")

# Function to load and process data
def load_and_process_data(url):
    """Loads web data, splits text, creates embeddings, and stores in FAISS."""
    loader = WebBaseLoader(url)
    documents = loader.load()

    # Split long documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Store split documents for debugging
    st.session_state.documents = chunks

    # Create FAISS vector store
    st.session_state.vector_db = FAISS.from_documents(chunks, embedding_model)

    st.success(f"Loaded {len(chunks)} document chunks and indexed them successfully!")

def get_answer(query):
    """Retrieves relevant context and generates an answer."""
    if "vector_db" not in st.session_state or st.session_state.vector_db is None:
        return "Please load data first."

    # Retrieve top 3 most relevant document chunks
    docs = st.session_state.vector_db.similarity_search(query, k=3)

    if not docs:
        return "No relevant information found in the loaded data."

    # Extract relevant content
    context = "\n".join([doc.page_content for doc in docs])

    # Debug retrieved context
    st.write("**Retrieved Context:**", context)

    # Define prompt with retrieved context
    full_prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant. Use the provided context to answer the question accurately.'),
        ('user', f'Context:\n{context}\n\nQuestion: {query}')
    ])

    output_parser = StrOutputParser()
    chain = full_prompt | llm | output_parser
    return chain.invoke({'question': query})

# Streamlit UI
st.title("AI Chatbot with Context Learning")

# Sidebar for data loading
st.sidebar.header("Load Data")
url = st.sidebar.text_input("Enter webpage URL")
if st.sidebar.button("Load & Process Data"):
    load_and_process_data(url)

# Debug: Show stored documents
if "documents" in st.session_state:
    st.sidebar.write(f"Loaded {len(st.session_state.documents)} text chunks")

# Chat Interface
st.subheader("Chat with the AI")
input_text = st.text_input("Ask a question:")
if st.button("Get Answer"):
    answer = get_answer(input_text)
    st.write("**AI Response:**", answer)
