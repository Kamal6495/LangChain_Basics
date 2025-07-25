#closed Source Paid
#OpenAI account has no remaining credits for API usage. This happens when:

# Your free trial credits are exhausted.

# You are on a free plan with no billing set up.

# You reached the monthly quota limit on your paid plan.

###################################
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ['HF_HOME'] = "D:/Practice/Thesis/3_LangChainBasics/2.ChatModels"


embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

####Single Line####
result=embedding.embed_query("Delhi is capital of india")
print(str(result))


####Multi Line Docs####
document=["Delhi is capital of india ","Delhi hosted 2010 commonwealth games","India won t20 world cup"]
result1=embedding.embed_documents(document)
print(str(result1))


