from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.messages import HumanMessage
import os

# Set Proxy (if needed)
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Initialization
model = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# Output Parsers
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negetive'] = Field(description='Give feedback from the sentiment')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt Template
prompt1 = PromptTemplate(
    template="Classify the sentiment in postive or negetive from a following feedback text \n {feedback} \n{format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

# Build Chains
classifier_chain = prompt1 | model | parser2  # Forces structured output

# Use HumanMessage for feedback
custom_feedback = HumanMessage(content="""
There are several moments during this 104-minute film that you feel your stomach tightening and clamp your eyes shut in anticipation of what is about to unfold in Tumbbad.

A blend of folklore and fantasy, director Rahi Anil Barve’s imaginative horror (inspired by Marathi genre writer Narayan Dharap) builds on the mythology of Hastar—a god disgraced for this insatiable greed for wealth and food. However, in the perennially rain-soaked village of Tumbbad, Hastar is revered.

The film is divided into three chapters. Part one opens in 1918. Barve and co-director Adesh Prasad waste no time establishing the dark, wet and strange atmosphere. A widow and her two sons Vinayak and Sadashiv live in a secluded house. But there is a fourth resident and it is their task to tend to the rotting old woman chained in a dungeon, whose wrath can be managed by feeding her on time.

The immortal great-grandmother is the only one who knows where Hastar’s treasure is buried, and Vinayak is obsessed with discovering its whereabouts.

The high point of part one is the scene of the grandmother dragging the impudent Vinayak through the cavernous house. It’s everything this genre feature should be—scary, grisly and full of evil dread.

Chapter 2 moves forward 15 years. Vinayak (Sohum Shah), a roguish grown man living in Pune, is seen heading back to Tumbbad. This could have been an electric mid-section had it simply been about Vinayak reuniting with his great grandmother and the treasure hunt. But Barve pads it out with glimpses into Vinayak’s life as an affluent man with a wandering eye who is being exploited by Raghav (Deepak Damle), a local merchant.

Through the three chapters we see Vinayak’s growing greed, his focus on finding the treasure and obsession with possessions. Part 3 jumps ahead to 1947. Even as the British colonisers are preparing to exit India, Vinayak’s son (Mohammad Samad, with perhaps the best performance in the film), is showing hints of having inherited his father’s avarice. As he is being trained to become the next treasure-hunter, can things end well for a family hungry for wealth?

Tumbbad is eerie, imaginatively designed, stunningly filmed and well directed. Cinematographer Pankaj Kumar uses close ups and tight frames to simulate claustrophobia. Production designers Nitin Zihani Choudhury and Rakesh Yadav paint pulsating wombs, create dimly lit passages and wild overgrown trees. The stunning visual effects and creepy creature designs complement the art direction. Jesper Kyd’s music underscores the sense of foreboding that laces the entire saga written by Mitesh Shah, Anand Gandhi, Prasad and Barve and carefully edited by Sanyukta Kaza.
""")

# Invoke classifier chain
result = classifier_chain.invoke({'feedback': custom_feedback.content})
print("Sentiment:", result.sentiment)

# Summary Prompts
prompt2 = PromptTemplate(
    template="Give appropriate summary 3 lines on what is positive in feedback \n {feedback}",
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template="Give appropriate summary 3 lines on what is negative in feedback \n {feedback}",
    input_variables=['feedback']
)

# Branch logic
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negetive', prompt3 | model | parser),
    RunnableLambda(lambda x: "No appropriate response")
)

# Combine chains
chain = classifier_chain | branch_chain
result = chain.invoke({'feedback': custom_feedback.content})
print("Summary:", result)


chain.get_graph().print_ascii()