from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

import os
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

model=ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

template=PromptTemplate(
    template="Generate physics facts on {topic}",
    input_variables=['topic']
)
template2=PromptTemplate(
    
  template="Generate 7 line summary of {text}",
    input_variables=['text']
)

parser=StrOutputParser()
chain=template | model | parser | template2 |model | parser
result=chain.invoke({"topic":"The Lorentz Transformation is the mathematical core of Albert Einstein's theory of Special Relativity. It describes how measurements of space and time by two different observers are related."})
print(result)

chain.get_graph().print_ascii()