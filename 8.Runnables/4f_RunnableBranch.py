from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.8
)

# Prompts
prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text:\n{text}",
    input_variables=["text"] 
)

parser = StrOutputParser()

# Report generation chain
report_gen_chain = prompt1 | llm | parser

# Branching chain
branch_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 250,
        (prompt2| llm | parser),  # ✅ map string → {text}
    ),
    RunnablePassthrough()  # Default: return original report
)

# Final chain
final_chain = report_gen_chain | branch_chain

# Run
result = final_chain.invoke({"topic": "NASA"})
print(result)
