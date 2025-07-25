from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# os.environ["HTTP_PROXY"]="http://edcguest:edcguest@172.31.102.14:3128"
# os.environ["HTTPS_PROXY"]="http://edcguest:edcguest@172.31.102.14:3128"

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

chat_model=ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-lite',
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# Static prompt
prompt = "Summary of windows powershell in 2 line lines"

response = chat_model.invoke(prompt)
print(response.content)