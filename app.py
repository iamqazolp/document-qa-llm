from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_engine import generation_mode

#Initialize API
app = FastAPI(title="Document RAG Agent API", description="Internal Document Querying API")

#define mode: 'local' or 'cloud', default is 'cloud'
mode='cloud'

#Initialize RAG pipeline before handling requests to load model into RAM
generation=generation_mode[mode]

#Define input structure
class QueryRequest(BaseModel):
    query: str

class ModeRequest(BaseModel):
    new_mode: str
#get health
@app.get("/health")
async def health_check():
    return {"status": "healthy", "generation_mode": mode}

# FastAPI Endpoint
@app.post("/ask")
async def ask_document(req: QueryRequest):
    try:
        result = generation(req.query)
        
        return {
            "status": "success",
            "question": req.query,
            "answer": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System error: {str(e)}")    
@app.put("/change_mode")
async def change_mode(req: ModeRequest):
    global mode, generation
    if req.new_mode not in generation_mode:
        raise HTTPException(status_code=400, 
                            detail=f"Mode '{req.new_mode} is not supported.\n Please choose mode 'local' or 'cloud'!") 
    try:
        mode = req.new_mode
        generation=generation_mode[mode]
        return {
            "status": "success",
            "current_mode": mode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System error: {str(e)}")