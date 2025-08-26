from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM setup
llm = ChatGoogleGenerativeAI(              # Initialize Gemini LLM
    model="gemini-2.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.8
)

# Prompts
prompt1 = PromptTemplate(                  # Prompt for character names + description
    template="Write 10 characters names with some description of the video game topic:\n {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(                  # Prompt for reviews
    template="Write 5 best review of the game topic:\n\n{topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()                 # Parse output as plain string

# Word Counter Function
def word_counter(text):                    # Counts words in a string
    return len(text.split())

# Convert Python function to Runnable
runnable_word_count = RunnableLambda(word_counter)

# Chain for generating character details
character_chain = prompt1 | llm | parser

# --- First Way: using a named function ---
parallel_chain1 = RunnableParallel({
    "Character Details": RunnablePassthrough(),       # Keep raw output
    "Count Words": RunnableLambda(word_counter)       # Use word_counter() function
})

# --- Second Way: using inline lambda ---
parallel_chain2 = RunnableParallel({
    "Character Details": RunnablePassthrough(),       # Keep raw output
    "Count Words": RunnableLambda(lambda x: len(x.split()))  
    # lambda x: len(x.split()) means:
    #    lambda = anonymous (nameless) function
    #    x = the input it receives
    #    len(x.split()) = return the number of words in x
    #same as word_counter(x), but written inline
})

# Connect character_chain → parallel_chain
final_chain1 = character_chain | parallel_chain1
final_chain2 = character_chain | parallel_chain2

# Run both chains
result1 = final_chain1.invoke({'topic':'Crysis 2'})   # Input for first way
# print(result1)

result2 = final_chain2.invoke({'topic':'Crysis 2'})   # Input for second way
# print(result2)

# final_result="""{} \n word count: {}""".format(result1['Character Details'],result1['Count Words'])

final_result = f"{result1['Character Details']} \n word count: {result1['Count Words']}"

print(final_result)
