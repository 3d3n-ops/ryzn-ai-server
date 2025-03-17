# main.py

import os 
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
from typing import Optional
from pydantic import BaseModel
import pytesseract
from pdfminer.high_level import extract_text
from gtts import gTTS
from langchain_groq import ChatGroq
app = FastAPI()

# CORS middleware to allow requests from the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for uploads and static files
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static directory for serving audio files
app.mount("/static", StaticFiles(directory="static"), name="static")


groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY envinronment variable is not set")

# Initialize LLM
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama3-70b-8192"  # You can also use "mixtral-8x7b-32768" or other available models
)
class SummaryResponse(BaseModel):
    summary: str
    audio_url: Optional[str] = None

@app.post("/api/summarize", response_model=SummaryResponse)
async def summarize_document(
    file: UploadFile = File(...),
    output_type: str = Form("text")
):
    try:
        # Generate a unique file ID
        file_id = str(uuid.uuid4())
        
        # Create a path for the uploaded file
        file_extension = os.path.splitext(file.filename)[1]
        file_path = f"uploads/{file_id}{file_extension}"
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract text from file
        text = extract_text(file_path)
        
        # Generate summary
        summary = generate_summary(text)
        
        # Generate audio if requested
        audio_url = None
        if output_type == "audio":
            audio_filename = f"{file_id}.mp3"
            audio_path = f"static/{audio_filename}"
            
            tts = gTTS(text=summary, lang="en")
            tts.save(audio_path)
            
            audio_url = f"/static/{audio_filename}"
        
        return SummaryResponse(summary=summary, audio_url=audio_url)
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    
    except FileNotFoundError as fnfe:
        raise HTTPException(status_code=404, detail=str(fnfe))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


def extract_from_pdf(file_path):
    try:
        text = extract_text(file_path)
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

def extract_text(file_path):
    # Handle different file formats
    if file_path.lower().endswith(".pdf"):
        return extract_from_pdf(file_path)
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

async def generate_summary_stream(text):
    truncated_text = text[:6000]
    prompt = f"Summarize the following text: {truncated_text}"
    
    try:
        # Enable streaming
        from fastapi.responses import StreamingResponse
        
        async def stream_response():
            async for chunk in llm.astream(prompt):
                yield chunk.content
                
        return StreamingResponse(stream_response(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)