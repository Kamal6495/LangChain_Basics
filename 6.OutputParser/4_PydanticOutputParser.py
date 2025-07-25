from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ["HTTP_PROXY"]="http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"]="http://edcguest:edcguest@172.31.102.14:3128"
load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

model=ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

class Book(BaseModel):
    title:str=Field("Write Title of the book")
    synopsis:str=Field("Write the synopsis of book in 10 sentences with moral")
    author:list[str]=Field("Write name of Authors")
    pages:int=Field("Number of pages in book")

parser=PydanticOutputParser(pydantic_object=Book)

template=PromptTemplate(
    template="generate the best book on {topic} with title,synopsis,authors and pages \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
    )

#First Way
prompt=template.invoke({'topic':'Mathematics Illusions'})
print(prompt)
result=model.invoke(prompt)
finalresult=parser.parse(result.content)
print(finalresult)

#Second Way
# chain=template | model | parser
# result=chain.invoke({'topic':'Particle Physics'})
# print(result)
