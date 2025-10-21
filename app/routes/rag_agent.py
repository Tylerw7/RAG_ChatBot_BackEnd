from fastapi import APIRouter, HTTPException
from typing import TypedDict, Annotated
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings





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








