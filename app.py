import os
import subprocess
import chainlit as cl
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

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
    sources = [doc.metadata.get('source', 'Unknown') for doc in similar_docs]

    response = rag_chain.invoke(query)
    return response, sources

# Chainlit Event Handler
@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()
    if not query:
        await cl.Message(content="Query cannot be empty.").send()
        return

    response, sources = perform_rag_search(query)
    
    # Send response to the user
    await cl.Message(content=response).send()

    # Send sources if available
    if sources:
        sources_text = "\n".join(f"- {source}" for source in sources)
        await cl.Message(content=f"**Sources:**\n{sources_text}").send()

if __name__ == "__main__":
    import chainlit as cl

    subprocess.run(["chainlit", "run", "main.py"])
