import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables
load_dotenv()

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request model
class QueryRequest(BaseModel):
    query: str

# Initialize OpenAI Embeddings
embedding = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Retrieve existing indexes
existing_indexes = [index.name for index in pc.list_indexes()]
if not existing_indexes:
    raise ValueError("No Pinecone index found.")

# Select the first available index
index = pc.Index(existing_indexes[0])
print("Index Stats:", index.describe_index_stats())

# Initialize OpenAI Chat Model
openai_llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL"),
)

# Initialize Pinecone Vector Store
vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding,
)

# Define RAG Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert assistant providing detailed information about the Ballerina programming language.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Provide a comprehensive and precise coding answer based on the given context. If the context does not contain sufficient information,
    clearly state that you cannot provide a complete answer based on the available information.
    """
)

# Define Function to Format Documents for Retrieval
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG Chain
rag_chain = (
    {"context": vector_store.as_retriever() | format_docs, "question": RunnablePassthrough()}
    | prompt
    | openai_llm
    | StrOutputParser()
)

# Function to Perform RAG Search
def perform_rag_search(query):
    print("Retrieved Documents:")
    similar_docs = vector_store.similarity_search(query, k=3)  # Retrieve top 3 similar documents
    sources = []
    for doc in similar_docs:
        source = doc.metadata.get('source', 'Unknown')
        sources.append(source)
    
    response = rag_chain.invoke(query)
    return response, sources

# FastAPI Endpoint
@app.post("/query")
async def query_rag(data: QueryRequest):
    if not data.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    response, sources = perform_rag_search(data.query)
    return {"query": data.query, "response": response, "sources": sources}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)