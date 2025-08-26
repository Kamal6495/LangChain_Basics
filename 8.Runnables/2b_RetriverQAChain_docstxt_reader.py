#Why Runnables Needed
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA # Deprecated import
from langchain.chains.retrieval_qa.base import RetrievalQA  #New import

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

import os
load_dotenv()  # Load .env file

# Set proxy (needed in your network)
os.environ["HTTPS_PROXY"]="http://edcguest:edcguest@172.31.100.25:3128"
os.environ["HTTP_PROXY"]="http://edcguest:edcguest@172.31.100.25:3128"

# Paths
BASE_DIR=os.path.dirname(os.path.abspath(__file__))   # Current script dir
FILE_PATH=os.path.join(BASE_DIR,"docs.txt")          # docs.txt path

# API key
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

# LLM
model=ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',      # Gemini model
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2              # Control creativity
)

# Load and split docs
loader=TextLoader(FILE_PATH,encoding="utf-8")        # Load file
documents=loader.load()
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50                 # Split into chunks
)
docs=text_splitter.split_documents(documents)

# Vector DB
vectorstore=FAISS.from_documents(
    docs,
    GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",                # Embedding model
        google_api_key=GOOGLE_API_KEY
    )
)

# Retriever + QA Chain
retriever = vectorstore.as_retriever()               # Retriever interface
qa_chain = RetrievalQA.from_chain_type(
    llm=model, retriever=retriever                   # Create QA chain
)

# Query
topic=input("Enter Topic: ")                         
result=qa_chain.run(topic)   # .run() deprecated → use .invoke()

# Output
print("Blog\n--------------------\n",result)
