#Why Runnables Needed
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# ✅ Load environment variables from the .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Initialize Google Generative AI model
# - model: 'gemini-2.5-pro' (latest high-quality version)
# - temperature: 0.2 for more factual and stable responses
model = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# ✅ Define prompt template
# - Use {topic} placeholder so it can be replaced dynamically
prompt = PromptTemplate(
    template="Write a 50-line blog on the given topic:\n{topic}",
    input_variables=['topic']
)

# ✅ Take user input for the topic
topic = input("Enter Topic: ")

# ✅ Format the prompt (instead of .invoke())
# - prompt.format() returns a plain string, which the model expects
# - This avoids errors from passing a PromptValue object
formatted_prompt = prompt.format(topic=topic)

# ✅ Generate the blog using the AI model
# - model.invoke() can take plain text and return a response object
blog = model.invoke(formatted_prompt)

# ✅ Display the result
# - Some model outputs have `.content` attribute (chat format), others are plain strings
# - Use hasattr() to handle both cases safely
print("\nGenerated Blog:\n")
print(blog.content if hasattr(blog, "content") else blog)
