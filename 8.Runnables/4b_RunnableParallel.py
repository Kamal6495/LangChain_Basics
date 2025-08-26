from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel,RunnableSequence
import os

# Proxy settings
os.environ["HTTP_PROXY"]="http://edcguest:edcguest@172.31.100.25:3128"
os.environ["HTTPS_PROXY"]="http://edcguest:edcguest@172.31.100.25:3128"

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.8
)

# Prompts
prompt1 = PromptTemplate(
    template="Write 10 chararcters names of video game topic:\n {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Write 5 best thoughts of the topic:\n\n{topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()
#Parllel Chain (legacy style)
# parallel_chain = RunnableParallel({
#     "Historical Context": RunnableSequence(prompt1 , llm , parser),
#     "Historical Figures": RunnableSequence(prompt2 , llm , parser)
# })

# Parallel chain (modern style)
parallel_chain = RunnableParallel({
    "Historical Context": prompt1 | llm | parser,
    "Historical Figures": prompt2 | llm | parser
})

# Run
result = parallel_chain.invoke({"topic": "Crysis 2"})
print(result)
