import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

os.environ['LANGCHAIN_API_KEY']=''#Use your langchain api key
os.environ['LANGCHAIN_TRACKING_V2']='True'
os.environ['LANGCHAIN_PROJECT']='default'

prompt=ChatPromptTemplate.from_messages(
    [
        ('system','you are a helpful assistant.Please respond to the question asked'),
        ('user','Question:{question}')
    ]
)

##Streamlit  framework
st.title('LangChain Demo With Gemma2')
input_text=st.text_input('What question you have in mind?')

llm=Ollama(model='gemma:2b')

output_praser=StrOutputParser()
chain=prompt|llm|output_praser

if input_text:
    st.write(chain.invoke({'question':input_text}))