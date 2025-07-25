#ChatHuggingFace is not compatible
# ChatHuggingFace expects a chat-completion interface,
#  but HuggingFacePipeline provides text-generation. This mismatch will cause unexpected behavior or errors.

from langchain_huggingface import HuggingFacePipeline
import os
# os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
# os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ['HF_HOME'] = "D:/Practice/Thesis"

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Free model
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

response = llm.invoke("What is Physics in 3 lines")

# If it's a string, print directly
print("Response:", response)
print("Type:", type(response))

