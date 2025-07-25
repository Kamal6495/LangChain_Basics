from pydantic import BaseModel,EmailStr,Field
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

#Schema
class Review(BaseModel):
    key_themes:list[str]=Field(description="Write down all the key themes as discussed in review in a list")         
    summary:str=Field(description="Write Summary of the review concisely")
    sentiment:Literal["Pos","Neg"]=Field(description="Return sentiment of review either Positive or Negetive")
    pros:Optional[list[str]]=Field(default=None,description="write all the advantages of product in review")
    cons:Optional[list[str]]=Field(default=None,description="write all the disadvantages of product in review")
    


structured_model=model.with_structured_output(Review)
result=structured_model.invoke("""The Realme GT2 disappoints in several aspects despite initial hype. Its so-called “paper-like” back feels gimmicky rather than premium, and the design choice doesn’t justify its environmental claim. While it boasts a 120Hz AMOLED display and Snapdragon 888 processor on paper, the real-world experience is underwhelming due to significant overheating during gaming and prolonged use.

Battery life, although marketed as long-lasting, struggles under heavy usage, and the 65W fast charging seems more like a necessity than a luxury given the rapid battery drain. Camera performance is another letdown—the ultrawide lens delivers mediocre results, and the selfie camera performs poorly, especially in low light, leaving photos soft and lacking detail. Software improvements are minimal with Realme UI 3.0, which still carries unnecessary bloat.

Overall, the GT2 feels like a compromise-packed device, where flashy specs cannot hide the persistent heating issues, average cameras, and questionable design choices. Not recommended for users expecting sustained performance or reliable photography.

""")

print(result.sentiment)
