from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

import os

os.environ["HTTP_PROXY"]="http://edcguest:edcguest@172.31.100.25:3128"
os.environ["HTTPS_PROXY"]="http://edcguest:edcguest@172.31.100.25:3128"
load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

prompt1=PromptTemplate(
    template="Write 10 line historical paragraph on topic. \n{topic}",
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="Write important figures in the historical context from following: {text}",
    input_variables=['text']
)

parser=StrOutputParser()

#In LangChain 0.2+, you usually chain with | operator, not by passing all args to RunnableSequence.Instead of:
# chain=RunnableSequence(prompt,llm,parser) %Legacy%
chain=prompt1 | llm | prompt2 | llm | parser  #%Modern%

result=chain.invoke({'topic':'Egyptian'})
print("Author:", "Kamal Kant Singh")
print("-" * 10)
print(result)





