# OpenAI LLM class in langchain_openai uses the Completions API, which only supports models 
# like text-davinci-003 and gpt-3.5-turbo-instruct.

# These models are deprecated and require quota (which you don’t have).

# Modern OpenAI models (GPT-4o, GPT-4o-mini) do NOT support Completions API, only Chat API.
from langchain_openai import OpenAI
from dotenv import load_dotenv

import os
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"

load_dotenv()

llm=OpenAI(model='gpt-3.5-turbo-instruct')
result=llm.invoke("What is capital of Uttar Pradesh, India")
print(result)
