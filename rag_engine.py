
import chromadb
import ollama
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os


load_dotenv()
DIRECTORY_PATH = os.getenv("DIRECTORY_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DATABASE_PATH = os.getenv("DATABASE_PATH")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

ollama_client = ollama.Client(host=OLLAMA_HOST)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=DATABASE_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
groq_client = Groq(
    api_key=GROQ_API_KEY,
)


#Yayy we're not at the Retrieval step (the R in RAG)
def retrieval(user_query):
    #Convert the question into an embedding using the exact same model we used earlier
    #model.encode returns a numpy array, but Chroma expects a list, so we use .tolist()
    query_embedding = embedding_model.encode(user_query).tolist()

    #Query the ChromaDB collection
    n_result=10
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_result,
    )

    #Display the results clearly
    #Chroma returns data in lists of lists to handle multiple queries at once, 
    #so we access the first element [0] of the documents, metadatas, and distances
    return results['documents'][0]

def get_prompt(context_text):
    prompt = f"""
    You are a helpful and accurate assistant. Answer the user's question based ONLY on the provided context below. 
    If the context does not contain the answer, politely say "I don't have enough information in my knowledge base to answer that." 
    Do not use outside knowledge.

    Context:
    {context_text}
    """
    return prompt

#THE G IN RAG. HOORAY
def generation_cloud(user_query):
    context_text = "\n\n---\n\n".join(retrieval(user_query))
    prompt=get_prompt(context_text)
    
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
def generation_local(user_query):
    context_text = "\n\n---\n\n".join(retrieval(user_query))
    prompt=get_prompt(context_text)
    
    # Call your local Ollama server instead of Groq
    response = ollama_client.chat(
        model='llama3.2:3b', # Make sure this matches the model you downloaded
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": user_query,
            }
        ]
    )
    
    # Ollama's response structure is slightly different from Groq's
    return response['message']['content']

generation_mode = {'cloud': generation_cloud,
                   'local': generation_local}



