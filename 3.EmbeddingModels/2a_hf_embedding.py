from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.decomposition import PCA
import os

# Load environment variables
load_dotenv()

# Proxy and Hugging Face cache
# os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
# os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HF_HOME"] = "D:/Practice/Thesis"

# Initialize Hugging Face embeddings
hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Get embedding for a query
text = "Delhi is the capital of India"
result = hf_embed.embed_query(text)
print("Original Dim:", len(result))  # 384
print("Embedding of text : ",result)

