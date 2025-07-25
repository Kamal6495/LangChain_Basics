#closed Source Paid
#OpenAI account has no remaining credits for API usage. This happens when:

# Your free trial credits are exhausted.

# You are on a free plan with no billing set up.

# You reached the monthly quota limit on your paid plan.

###################################
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import os
load_dotenv()
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ['HF_HOME'] = "D:/Practice/Thesis/3_LangChainBasics/2.ChatModels"


embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

document=["Delhi, officially the National Capital Territory (NCT) of Delhi, is a city and a union territory of India containing New Delhi, the capital of India.",
            "Straddling the Yamuna river, but spread chiefly to the west, or beyond its right bank, Delhi shares borders with the state of Uttar Pradesh in the east and with the state of Haryana in the remaining directions.",
            "Delhi became a union territory on 1 November 1956 and the NCT in 1995.",
            "The NCT covers an area of 1,484 square kilometres (573 sq mi).",
            "According to the 2011 census, Delhi's city proper population was over 11 million, while the NCT's population was about 16.8 million.",
            "The topography of the medieval fort Purana Qila on the banks of the river Yamuna matches the literary description of the citadel Indraprastha in the Sanskrit epic Mahabharata; however, excavations in the area have revealed no signs of an ancient built environment.",
            "From the early 13th century until the mid-19th century, Delhi was the capital of two major empires, the Delhi Sultanate and the Mughal Empire, which covered large parts of South Asia.",
            "All three UNESCO World Heritage Sites in the city, the Qutub Minar, Humayun's Tomb, and the Red Fort, belong to this period.",
            "Delhi was the early centre of Sufism and Qawwali music.",
            "The names of Nizamuddin Auliya and Amir Khusrau are prominently associated with it.",
            "The Khariboli dialect of Delhi was part of a linguistic development that gave rise to the literature of Urdu and later Modern Standard Hindi.",
            "Major Urdu poets from Delhi include Mir Taqi Mir and Mirza Ghalib.",
            "Delhi was a notable centre of the Indian Rebellion of 1857.",
            "In 1911, New Delhi, a southern region within Delhi, became the capital of the British Indian Empire.",
            "During the Partition of India in 1947, Delhi was transformed from a Mughal city to a Punjabi one, losing two-thirds of its Muslim residents, in part due to the pressure brought to bear by arriving Hindu and Sikh refugees from western Punjab.",
            "After independence in 1947, New Delhi continued as the capital of the Dominion of India, and after 1950 of the Republic of India.",
            "Delhi's urban agglomeration, which includes the satellite cities Ghaziabad, Faridabad, Gurgaon, Noida, Greater Noida and YEIDA city located in an area known as the National Capital Region (NCR), has an estimated population of over 28 million, making it the largest metropolitan area in India and the second-largest in the world (after Tokyo).",
            "Delhi ranks fifth among the Indian states and union territories in human development index, and has the second-highest GDP per capita in India (after Goa).",
            "Although a union territory, the political administration of the NCT of Delhi today more closely resembles that of a state of India, with its own legislature, high court and an executive council of ministers headed by a chief minister.",
            "New Delhi is jointly administered by the federal government of India and the local government of Delhi, and serves as the capital of the nation as well as the NCT of Delhi.",
            "Delhi is also the centre of the National Capital Region, which is an \"interstate regional planning\" area created in 1985.",
            "Delhi hosted the inaugural 1951 Asian Games, the 1982 Asian Games, the 1983 Non-Aligned Movement summit, the 2010 Men's Hockey World Cup, the 2010 Commonwealth Games, the 2012 BRICS summit, the 2023 G20 summit, and was one of the major host cities of the 2011 and 2023 Cricket World Cups."   
]

query="Partition of India in which year?"

#Document/Query Embedding findinding
doc_embedding=embedding.embed_documents(document)
query_embedding=embedding.embed_query(query)

print(cosine_similarity([query_embedding],doc_embedding)) #[query_embedding] convert to 2d list

scores=cosine_similarity([query_embedding],doc_embedding)[0]# 2d to 1d
index,score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(document[index])
print(score)