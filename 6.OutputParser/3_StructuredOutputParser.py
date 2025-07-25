from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Set Proxy (Optional)
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Model
model = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2  # ✅ fixed typo
)

# Define Response Schema
schema = [
    ResponseSchema(name='Book 1', description='Name of First Book'),
    ResponseSchema(name='Book 2', description='Name of Second Book'),
    ResponseSchema(name='Book 3', description='Name of Third Book')
]

# Output Parser
parser = StructuredOutputParser.from_response_schemas(schema)

# Prompt Template
template = PromptTemplate(
    template='Give 3 best books on {topic}.\n{format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# Create Chain
chain = template | model | parser

# Invoke
result = chain.invoke({'topic': 'Electromagnetic Theory of light'})

print(result)
