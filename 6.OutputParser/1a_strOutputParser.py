#Not Using StrOutputParser Using Template For Output (Why use output parser)
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

import os

# ✅ Set Proxy before anything else
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"


# ✅ Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

template1=PromptTemplate(
    template='Write detailed report on {topic}',
    input_variables=['topic']
)

template2=PromptTemplate(
    template='Write 5 Line summary on following text.\n{text}',
    input_variables=['text']
)


prompt1=template1.invoke({'topic':'Three Body Problem'})
result1=model.invoke(prompt1)

prompt2=template2.invoke({'text':result1.content})
result2=model.invoke(prompt2)

print(result2.content)