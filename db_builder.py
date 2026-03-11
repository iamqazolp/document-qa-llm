
# Inspired by _[this article](https://medium.com/red-buffer/building-retrieval-augmented-generation-rag-from-scratch-74c1cd7ae2c0)_

import fitz  #PyMuPDF
import os
import uuid
from dotenv import load_dotenv


load_dotenv()
DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DATABASE_PATH = os.getenv("DATABASE_PATH")


def extract_text_pdf(file_path: str):
    if not file_path.endswith('pdf'):
        raise Exception('File extension must be pdf!')
    pdf = fitz.open(file_path)
    pdf_texts = [page.get_text('text') for page in pdf]
    pdf.close()
    return ' '.join(pdf_texts)
        

def extract_text_txt(file_path: str):
    if not file_path.endswith('txt'):
        raise Exception('File extension must be txt!')
    with open(file_path,'r',encoding='utf-8') as file:
        return file.read()

extension_handler={'pdf': extract_text_pdf,'txt': extract_text_txt}

'''problem: documents can include a noticable number of short paragraphs
of just a few lines or even a few characters. This can significantly increase 
the number of chunks'''
def merge_small_paragraphs(paragraphs: list, chunk_size):
    new_paragraphs=[]
    i=0
    for p in paragraphs:
        if (not new_paragraphs) or len(new_paragraphs[-1])>=chunk_size//2:
            new_paragraphs.append(p)
        elif len(new_paragraphs[-1])<chunk_size//2:
            new_paragraphs[-1]+=p
    return new_paragraphs
            

def chunking(document: str, metadata={}, chunk_size=256, overlap=36):
    if overlap>=chunk_size:
        raise Exception('Overlap must be smaller than chunk size!')
    chunks={}
    paragraphs=[paragraph.split() for paragraph in document.split('\n') if paragraph]
    paragraphs=merge_small_paragraphs(paragraphs,chunk_size)
    for paragraph in paragraphs:
        current_position=0
        while current_position<len(paragraph):
            chunk_id=str(uuid.uuid4())
            chunk_text=paragraph[current_position:min(current_position+chunk_size,len(paragraph))]
            chunks[chunk_id]={'text': ' '.join(chunk_text),
                'metadata' : metadata }
            current_position+=chunk_size-overlap
    return chunks
    

documents={}
for prefix,_,filenames in os.walk(DOCUMENTS_FOLDER):
    for filename in filenames:
        file_path=os.path.join(prefix,filename)
        extension=filename.split('.')[-1]
        if extension in extension_handler.keys():
            document_id=str(uuid.uuid4()) #uuid4: random id
            document=extension_handler[extension](file_path)
            documents[document_id]=chunking(document,{'filename':filename,
                                                      'document_id':document_id})

all_chunks={}
for chunks in documents.values():
    all_chunks.update(chunks)

from sentence_transformers import SentenceTransformer

# all-MiniLM-L6-v2 small, fast, and highly effective for semantic search
embedding = SentenceTransformer(EMBEDDING_MODEL)

chunk_ids = list(all_chunks.keys())
texts_to_embed = [all_chunks[chunk_id]['text'] for chunk_id in chunk_ids]

print(f"Generating embeddings for {len(texts_to_embed)} chunks...")

#Generate embeddings
embeddings = embedding.encode(texts_to_embed, show_progress_bar=True)

for i, chunk_id in enumerate(chunk_ids):
    #Model ouputs np arrays so we need to convert np array to list 
    all_chunks[chunk_id]['embedding'] = embeddings[i].tolist() 

first_chunk_id = chunk_ids[0]
embedding_length = len(all_chunks[first_chunk_id]['embedding'])

print(f"Embedding has {embedding_length} dimensions.")


import json
'''initial plan, but the resulting json file is over 100MB for just 5 short 
documents. Plus retrieval is going to be incredibly slow since we have to retrieve
every embedding, convert them from lists to np arrays and then calculate
the cosine similarity with user's query'''
def save_to_json(file_path ="knowledge_base.json"):
    with open(file_path, "w") as f:
        json.dump(all_chunks, f, indent=4)
    print(f"Successfully saved {len(all_chunks)} chunks to {file_path}")

def load_from_json(file_path):
    with open(file_path, "r") as f:
        loaded_chunks = json.load(f)
    return loaded_chunks
import numpy as np
def cosine_similarity(Array,Brray):
    return np.dot(Array,Brray)/np.linalg.norm(Array)/np.linalg.norm(Brray)
def retrieval(query: np.ndarray,chunks: dict):
    max_chunk_id=None
    max_similarity=-1
    for chunk_id in chunks:
        embedding=np.array(chunks[chunk_id]['embedding'])
        similarity=cosine_similarity(query, embedding)
        if similarity>max_similarity:
            max_similarity=similarity
            max_chunk_id=chunk_id
    return max_chunk_id


#Use ChomaDB to follow industry's standard
import chromadb


#Initialize a persistent client. 
#This will create a folder current directory
client = chromadb.PersistentClient(path=DATABASE_PATH)

def reset_db():
    client.delete_collection(name=COLLECTION_NAME)
    print("Collection deleted!")
reset_db()
#Create a collection (kinda like a table in a traditional db)
#get_or_create_collection so we dont create a new collection everytime we run this cell
collection = client.get_or_create_collection(name=COLLECTION_NAME)

#ChromaDB expects data in separate lists, so we need to unpack our all_chunks dictionary
chunk_ids = []
documents = []
metadatas = []
embeddings = []

for chunk_id, chunk_data in all_chunks.items():
    chunk_ids.append(chunk_id)
    documents.append(chunk_data['text'])
    metadatas.append(chunk_data['metadata'])
    embeddings.append(chunk_data['embedding']) # Using the generated embeddings
    
chunks_count=len(chunk_ids)
print(f"Preparing to load {chunks_count} chunks into ChromaDB...")

max_batch_size=client.get_max_batch_size()
print(f'Max batch size of client: {max_batch_size}')

#ChromaDB does not support modifying the max batch size so we recycle the chunking logic
pos=0
while pos<chunks_count:
    right_index=min(chunks_count,pos+max_batch_size)
    collection.upsert(
        ids=chunk_ids[pos:right_index],
        documents=documents[pos:right_index],
        metadatas=metadatas[pos:right_index],
        embeddings=embeddings[pos:right_index]
    )
    pos+=max_batch_size

print(f"Successfully stored {collection.count()} chunks in the database.")





