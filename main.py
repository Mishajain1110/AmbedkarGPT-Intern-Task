# Import necessary libraries
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Load the input document
loader = TextLoader("input.txt", encoding="utf-8")
doc = loader.load()

# Split the document into smaller chunks
char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Split the document
docs = char_text_splitter.split_documents(doc)

# Initialize ChromaDB and create a collection
client = chromadb.Client()
collection = client.create_collection(name="new_collection")

# Create embeddings using HuggingFace model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a vector store from the documents and embeddings
vectorstore = Chroma.from_documents(docs, embedding_model)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# Initialize the Ollama LLM 
model = Ollama(model="mistral")

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    return_source_documents=True
)

# Example query to test the setup
query = "What are the main points or key ideas?"

# Get the response from the LLM
response = qa_chain.invoke({"query": query})
print("Response by the LLM:", response["result"])
