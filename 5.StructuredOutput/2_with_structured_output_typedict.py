from typing import TypedDict, Optional, Annotated, Literal
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# ✅ Set Proxy before anything else
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"

# ✅ Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

# ✅ Schema using TypedDict
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes as discussed in review in a list"]
    summary: Annotated[str, "Write Summary of the review concisely"]
    sentiment: Annotated[Literal["Pos", "Neg"], "Return sentiment of review either Positive or Negative"]
    pros: Annotated[Optional[list[str]], "write all the advantages of product in review"]
    cons: Annotated[Optional[list[str]], "write all the disadvantages of product in review"]

# ✅ Enable structured output
structured_model = model.with_structured_output(Review)

# ✅ Input text
input_text = """
The realme GT2 offers a compelling package with its unique paper-like back, vibrant 120Hz display, and a powerful Snapdragon 888 processor, delivering excellent performance for gaming and everyday tasks. The 5000mAh battery and 65W charging ensure long-lasting power and quick refills. However, the ultrawide camera and selfie camera might be areas for improvement, and the phone's performance can be affected by the Snapdragon 888's tendency to overheat.

Key Features and Strengths:
Unique Design:
The GT2's paper-like back made from biopolymer material offers a distinctive and environmentally conscious design, according to Forbes and YouTube.
Powerful Performance:
The Snapdragon 888 processor provides smooth performance for demanding tasks like gaming and multitasking.
Excellent Display:
The 120Hz AMOLED display with 1,300 nits of peak brightness offers a vibrant and responsive visual experience.
Long-Lasting Battery:
The 5000mAh battery and 65W SuperDart charging provide ample power and quick charging capabilities.
Software and Updates:
The phone runs Android 12 with realme UI 3.0, offering a clean and relatively bloat-free experience, according to Moneycontrol and Notebookcheck.

Potential Downsides:
Ultrawide Camera:
The ultrawide camera's image quality is not as impressive as the main camera, according to GSMArena.com and XDA Forums.
Selfie Camera:
The selfie camera's performance, especially in low light, might not be as sharp or detailed as desired, according to YouTube.
Overheating:
The Snapdragon 888 processor can get hot during intense gaming or prolonged use
"""

# ✅ Invoke structured output
result = structured_model.invoke(input_text)

# ✅ Print specific field
print("Sentiment:", result["sentiment"])
print("Summary:", result["summary"])
print("Pros:", result["pros"])
print("Cons:", result["cons"])
