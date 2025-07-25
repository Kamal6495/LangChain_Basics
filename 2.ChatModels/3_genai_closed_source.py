# A closed-source chat model is:

# ✔ A proprietary large language model (LLM) that is not open source, meaning its architecture, training data, and weights are not publicly available.
# ✔ You can only access it via an API (or through a company’s platform), not download or run locally.
# ✔ Examples:

# OpenAI GPT family (GPT-4o, GPT-4.1, GPT-3.5)

# Anthropic Claude (Claude 3 Opus, Sonnet, Haiku)

# Google Gemini (Gemini 2.5 Pro/Flash)



from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

import os
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-pro')
result=model.invoke("What is capital of Uttar Pradesh, India")
print(result.content)