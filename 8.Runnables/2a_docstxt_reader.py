#Why Runnables Needed
from dotenv import load_dotenv                   # Load .env variables
from langchain_core.prompts import PromptTemplate  # Prompt templates
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini model
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split docs
from langchain_community.vectorstores import FAISS  # Vector store
from langchain_community.document_loaders import TextLoader  # Load text docs
import os

# Proxy setup (if needed)
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.100.25:3128"   # HTTP proxy
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.100.25:3128"  # HTTPS proxy

# Load environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # Project base dir
FILE_PATH = os.path.join(BASE_DIR, "docs.txt")               # Input file path
load_dotenv()                                                # Load .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")                 # API key

if not GOOGLE_API_KEY:                                       # Validate key
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# Create Gemini model
model = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',          # Model name
    google_api_key=GOOGLE_API_KEY,   # Auth
    temperature=0.2                  # Lower = more factual
)

# Load and split docs
loader = TextLoader(FILE_PATH, encoding="utf-8")             # Load text file
documents = loader.load()                                    # Convert to docs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50                         # Split config
)
docs = text_splitter.split_documents(documents)              # Split docs

# Create embeddings + vector store
vectorstore = FAISS.from_documents(
    docs,
    GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",                        # Embedding model
        google_api_key=GOOGLE_API_KEY                        # Auth
    )
)
retriever = vectorstore.as_retriever()                       # Convert to retriever

# Retrieve relevant docs
query = "Arjun Chariot"                                      # Search query
retrieved_docs = retriever.get_relevant_documents(query)     # Get matches
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])  # Join text

# Prompt template
template = """
You are an expert writer.
Read the following Docs content and write a detailed 50-line blog
focused on the topic: "{topic}"

Docs content:
{docs_content}
"""
prompt_template = PromptTemplate(
    input_variables=["topic", "docs_content"], template=template  # Define template
)
final_prompt = prompt_template.format(topic=query, docs_content=retrieved_text)  # Fill template

# Invoke LLM
answer = model.invoke(final_prompt)                           # Call Gemini
print(answer)                                                 # Print result
