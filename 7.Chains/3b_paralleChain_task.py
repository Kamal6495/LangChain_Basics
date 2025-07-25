from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
import os

# Set proxy
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the model
model = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# Prompt templates
prompt1 = PromptTemplate(
    template="Generate a detailed review on the given topic: {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template="Generate the pros of the given review:\n{text}",
    input_variables=['text']
)
prompt3 = PromptTemplate(
    template="Generate the cons of the given review:\n{text}",
    input_variables=['text']
)

# Output parser
parser = StrOutputParser()

# Step 1: Create chain to generate main review
review_chain = prompt1 | model | parser

# Step 2: Create parallel chain for pros and cons
parallel_chain = RunnableParallel({
    'pros': prompt2 | model | parser,
    'cons': prompt3 | model | parser
})

# Step 3: Combine pros and cons into a single string
combine_chain = RunnableLambda(
    lambda x: f"Pros:\n{x['pros']}\n\nCons:\n{x['cons']}"
)

# Full pipeline: Generate review -> extract pros and cons -> combine
full_chain = review_chain | (lambda review: {'text': review}) | parallel_chain | combine_chain

# Run the chain with a topic
# result = full_chain.invoke({'topic': 'Atomization of Molecule'})
# print(result)

full_chain.get_graph().print_ascii()
