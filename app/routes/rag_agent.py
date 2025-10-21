from fastapi import APIRouter, HTTPException
from typing import TypedDict, Annotated
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma







# API KEY
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")



# Router
router = APIRouter(
    prefix='/chatbot',
    tags=['chatbot']
)

# --------------------- RAG ----------------------- #

### Embeddings
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key
)

### Load PDF
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "cbw_rag_info.pdf")
pdf_loader = PyPDFLoader(pdf_path)
pages = pdf_loader.load()


### Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

page_split = text_splitter.split_documents(pages)


### Vector Store
vectorstore = Chroma.from_documents(
    documents=page_split,
    embedding=embeddings,
    persist_directory="./chatbot_vector_store",
    collection_name="rag_chatbot"
)

### ---------------- Retriever Function --------------- ###
retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={"k":4}
)

result = retriever("what services do you provide?")

print(result)




