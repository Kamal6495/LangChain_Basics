# This will Not Work since hf_inference is not free it is paid ,sO PAY FOR API TO WORK
#ChatHuggingFace expects a chat-completion interface, 
# but HuggingFaceEndpoint (even with task="text-generation") does not provide that.
from dotenv import load_dotenv
from langchain_huggingface import  ChatHuggingFace,HuggingFaceEndpoint
import os

# Proxy setup
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"

# Load token from .env
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Free model
    task="text-generation",
    provider='auto'
)

model=ChatHuggingFace(llm=llm)


# Invoke model
response = model.invoke("What is the capital of Uttar Pradesh, India?")
print(response)
