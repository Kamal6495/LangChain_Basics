#With OutPut Parser

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

import os
#✅ Set Proxy before anything else
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
model=ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    google_api_key=GOOGLE_API_KEY,
    temparature=0.2
)

template1=PromptTemplate(
    template='Write detailed report on {topic}',
    input_variables=['topic']
)

template2=PromptTemplate(
    template='Write 5 Line summary on following text.\n{text}',
    input_variables=['text']
)

parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser
result=chain.invoke({'topic':'Three Body Problem'})

print(result)


