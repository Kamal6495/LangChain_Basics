#Why Runnables Needed
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

import os 
load_dotenv()

os.environ["HTTPS_PROXY"]="http://edcguest:edcguest@172.31.100.25:3128"
os.environ["HTTP PROXY"]="http://edcguest:edcguest@172.31.100.25:3128"

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
model=ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    model='gemini-2.5-pro'
)

prompt = PromptTemplate(
    template="Write a 50-sentence blog on the given topic:\n{topic}",
    input_variables=['topic']
)

chain=LLMChain(llm=model,prompt=prompt)
topic=input("Enter Topic:\n")
output=chain.run(topic)

print("Generated Bloag:\n-------------------------------------------------------------------------------\n",output)
