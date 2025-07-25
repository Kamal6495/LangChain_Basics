#With OutPut Parser

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser

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

parser=JsonOutputParser()



template=PromptTemplate(
    template = "Write detailed report on {topic}.\n {formatinstruction}",
    input_variables=['topic'],
    partial_variables={'formatinstruction':parser.get_format_instructions()}
)

prompt=template.invoke({'topic':'Algae-Single Celled'})
print(prompt)#LLM mein yehi prompt gaya hai

result=model.invoke(prompt)
finalres=parser.parse(result.content)
print(type(finalres))
print('\n')
print(finalres)

# chain=template|model|parser

# result=chain.invoke({'topic':'mars mission is feasible or not'})

# # print(result)
# print(type(result))




