from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough
import os

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM setup
llm = ChatGoogleGenerativeAI(             # Initialize Gemini LLM
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.8
)

# Prompts
prompt1 = PromptTemplate(                 # Prompt for character names
    template="Write 10 characters names of video game topic:\n {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(                 # Prompt for reviews
    template="Write 5 best review of the game topic:\n\n{topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()                  # Parse output as plain string

# Creating first chain
character_chain = prompt1 | llm | parser  # Generates character names

# Creating parallel chain for raw + reviews
parallel_chain = RunnableParallel({
    "Character Names": RunnablePassthrough(), # Passes through previous output (raw)
    "Best Reviews": prompt2 | llm | parser    # Generates reviews
})

# Connecting both chains
final_chain = character_chain | parallel_chain  # First run char_chain, then split to parallel

# Run chain
result = final_chain.invoke({'topic':'Crysis 2'})  # Input topic for both prompts
print(result)
