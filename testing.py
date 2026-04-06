
from groq import Groq
from dotenv import load_dotenv
import os


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
def get_prompt():
    prompt = f"""
    You are a sentiment analyst. Given the user's review of a movie, classify it as positive or negative.
    If positive, only answer a single character '1', else answer '0'.
    """
    return prompt
groq_client = Groq(
    api_key=GROQ_API_KEY,
)

def generation_cloud(user_query):
    prompt=get_prompt()
    
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": user_query,
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content
f=open('answers.txt','w')
import pandas as pd
import time
import tqdm
t=pd.read_csv('./data/test.csv')
for i in tqdm.tqdm(range(2000)):
    print(f'{generation_cloud(t.iloc[i,1])}\n',file=f)
f.close()