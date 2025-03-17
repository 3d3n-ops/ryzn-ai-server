# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
import logging
from typing import Optional, List
from pydantic import BaseModel
import pytesseract
from pdf2image import convert_from_path
from langchain_openai import ChatOpenAI
from gtts import gTTS
from dotenv import load_dotenv
import whisper  # OpenAI Whisper package
from google.cloud import texttospeech
from pathlib import Path
import re
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables first
load_dotenv()


## These are neccesary paths and configurations for the app to run. Tesseract helps with extracting the text from the file, 
## Poppler helps with converting the PDF to images, and FFmpeg helps with combining the audio files.
## Google Cloud credentials are used to generate the audio.
## The paths are set up for Windows.

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CREDENTIALS_DIR = BASE_DIR / "credentials"
CREDENTIALS_FILE = CREDENTIALS_DIR / "google-credentials.json"

# Ensure directories exist
CREDENTIALS_DIR.mkdir(exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Set paths for Windows
if os.name == 'nt':  # Windows
    # Tesseract configuration
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        logger.error(f"Tesseract not found at {pytesseract.pytesseract.tesseract_cmd}")
        raise RuntimeError(f"Tesseract not found at {pytesseract.pytesseract.tesseract_cmd}. Please install Tesseract and ensure it's in the correct location.")

    # Poppler configuration
    POPPLER_PATH = r"C:\Program Files\poppler-24.08.0\Library\bin"
    if not os.path.exists(POPPLER_PATH):
        logger.error(f"Poppler not found at {POPPLER_PATH}")
        raise RuntimeError(f"Poppler not found at {POPPLER_PATH}. Please install Poppler and ensure it's in the correct location.")

    # FFmpeg configuration
    FFMPEG_PATH = r"C:\Users\3d3n2\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin"  # Update this path to where you installed FFmpeg
    if not os.path.exists(FFMPEG_PATH):
        logger.error(f"FFmpeg not found at {FFMPEG_PATH}")
        raise RuntimeError(f"FFmpeg not found at {FFMPEG_PATH}. Please install FFmpeg and ensure it's in the correct location.")
    
    # Add FFmpeg to system PATH for pydub
    if FFMPEG_PATH not in os.environ['PATH']:
        os.environ['PATH'] += os.pathsep + FFMPEG_PATH

# Verify Google Cloud credentials
google_creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if not google_creds_path:
    logger.error("GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
    raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set in .env file. Please add it to your .env file.")

# Convert relative path to absolute if necessary
if not os.path.isabs(google_creds_path):
    google_creds_path = str(BASE_DIR / google_creds_path)

if not os.path.exists(google_creds_path):
    logger.error(f"Google Cloud credentials file not found at: {google_creds_path}")
    logger.error(f"Please place your google-credentials.json file in: {CREDENTIALS_DIR}")
    raise RuntimeError(f"Google Cloud credentials file not found. Please place your google-credentials.json file in: {CREDENTIALS_DIR}")

# Set the environment variable with the absolute path
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds_path
logger.debug(f"Using Google Cloud credentials from: {google_creds_path}")

app = FastAPI()

# CORS middleware to allow requests from the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for serving audio files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)
class ChatResponse(BaseModel):
    response: str
    user_query: str

class ChatRequest(BaseModel):
    text: str
    user_query: str

class SummaryResponse(BaseModel):
    summary: str
    audio_url: Optional[str] = None
    notes: Optional[str] = None
    tts_audio_url: Optional[str] = None
    transcript: Optional[str] = None
    quiz: Optional[str] = None


## This is the code for the summarize feature. User upload file and it will generate a summary of the file.
## The user can have a text summary of a file or an audio summary of a file.
## The user can also have notes generated from the file in markdown format.
## Finally, the user can have a tts audio version of the file that can be played. 

@app.post("/api/transcribe", response_model=SummaryResponse)
async def transcribe_document(audio_file: UploadFile = File(...)):
    pass
    try:
        audio_path = f"uploads/{audio_file.filename}"
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        transcript = transcribe_audio_file(audio_path)
        return SummaryResponse(transcript=transcript)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")



@app.post("/api/summarize", response_model=SummaryResponse)
async def summarize_document(
    file: UploadFile = File(...),
    output_type: str = Form("text")
):
    # Generate a unique file ID
    file_id = str(uuid.uuid4())
    file_path = None
    
    try:
        # Create a path for the uploaded file
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ['.pdf', '.txt']:
            raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF and TXT files are supported.")
        
        file_path = f"uploads/{file_id}{file_extension}"
        
        # Save the uploaded file
        logger.debug(f"Saving uploaded file to {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Extract text from file
        logger.debug("Starting text extraction")
        text = extract_text(file_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file.")
        logger.debug(f"Extracted text length: {len(text)}")
        
        # Generate summary
        logger.debug("Generating summary")
        summary = generate_summary(text)

        # Generate notes if requested
        notes = None
        if output_type == "notes":
            logger.debug("Generating notes")
            notes = generate_notes(text)    
        
        # Generate quiz if requested
        quiz = None
        if output_type == "quiz":
            logger.debug("Generating quiz")
            quiz = generate_quiz(text) 

        # Generate audio if requested
        audio_url = None
        if output_type == "audio":
            logger.debug("Generating audio")
            audio_filename = f"{file_id}.mp3"
            audio_path = f"static/{audio_filename}"
            
            tts = gTTS(text=summary, lang="en")
            tts.save(audio_path)
            
            audio_url = f"/static/{audio_filename}"

        # Generate audio reading from text if requested
        tts_audio_url = None
        if output_type == "tts_audio":
            try:
                logger.debug("Generating audio with Google Cloud Studio voice")
                audio_filename = f"{file_id}.mp3"
                audio_path = f"static/{audio_filename}"
                
                # Use the long audio synthesis function
                synthesize_long_audio(text, audio_path)
                tts_audio_url = f"/static/{audio_filename}"
                
            except Exception as e:
                logger.error(f"Error generating audio: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")
        
        return SummaryResponse(summary=summary, audio_url=audio_url, notes=notes, tts_audio_url=tts_audio_url, quiz=quiz)
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up the uploaded file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up file: {str(e)}")


@app.post("/api/response", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    try:
        # Truncate text to avoid token limits
        truncated_text = request.text[:4000]  # Limit to 4000 characters
        
        prompt = f"""You are a helpful assistant for students, helping them answer questions about the files they upload for studying. 
        With the context of the following text:

        {truncated_text}

        Answer the user's query: {request.user_query}
        
        Be concise and direct. Be clear when you don't know the answer based on the provided context."""
        
        logger.debug("Sending request to OpenAI")
        response = llm.invoke(prompt)
        generated_response = response.content
        
        return ChatResponse(
            response=generated_response,
            user_query=request.user_query
        )
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


def extract_text(file_path):
    # Handle different file formats
    if file_path.lower().endswith(".pdf"):
        logger.debug("Processing PDF file")
        return extract_from_pdf(file_path)
    elif file_path.lower().endswith(".txt"):
        logger.debug("Processing TXT file")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

def extract_from_pdf(file_path):
    # Convert PDF to images and extract text using OCR
    try:
        logger.debug("Converting PDF to images")
        if os.name == 'nt':  # Windows
            from pdf2image.exceptions import PDFPageCountError
            try:
                logger.debug(f"Using Poppler path: {POPPLER_PATH}")
                images = convert_from_path(
                    file_path,
                    poppler_path=POPPLER_PATH
                )
            except PDFPageCountError as e:
                logger.error(f"Error converting PDF: {str(e)}")
                raise HTTPException(status_code=500, detail="Invalid or corrupted PDF file")
            except Exception as e:
                logger.error(f"Error converting PDF with Poppler: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error converting PDF: {str(e)}")
        else:
            images = convert_from_path(file_path)
        
        logger.debug(f"Converted PDF to {len(images)} images")
        text = ""
        for i, image in enumerate(images):
            logger.debug(f"Processing image {i+1}/{len(images)}")
            try:
                page_text = pytesseract.image_to_string(image)
                text += page_text + "\n\n"
                logger.debug(f"Successfully extracted text from image {i+1}")
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing PDF page {i+1}: {str(e)}")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")
        return text
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error in extract_from_pdf: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def generate_summary(text):
    # Use LLM to generate summary
    # Truncate text to avoid token limits
    truncated_text = text[:4000]
    prompt = f"Summarize the following text in a concise and informative way: {truncated_text}"
    
    try:
        logger.debug("Sending request to OpenAI")
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

def generate_quiz(text):
    # Use LLM to generate summary
    # Truncate text to avoid token limits
    truncated_text = text[:4000]
    prompt = f"With the context of this text:{truncated_text}, generate a quiz for the user ranging from 5 to 20 questions to help the user practice and test their understanding of the concepts in the text. Present your response in markdown format "
    
    try:
        logger.debug("Sending request to OpenAI")
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating quiz: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")


def transcribe_audio_file(audio_path):
    # Use Whisper to transcribe the audio file
    try:
        # Load the Whisper model
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")

def generate_notes(text):
    # Use LLM to generate notes
    prompt = f"Generate notes from the following text: {text}. Use markdown format. Use the following format: # Title\n\n## Summary\n\n## Key Points\n\n## Additional Information\n\n## References"
    try:
        logger.debug("Sending request to OpenAI")
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating notes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating notes: {str(e)}")


## This is the code for the audio reading feature. User upload file and it will generate a audio file that can be played.
## It uses the Google Cloud Text-to-Speech API to generate the audio.
## It splits the text into chunks and combines them into a single audio file.
## It also has a timeout of 10 minutes.

def split_text_into_chunks(text: str, max_bytes: int = 4800) -> List[str]:
    """Split text into chunks that are within the byte limit."""
    chunks = []
    current_chunk = ""
    
    # Split by sentences (simple approach)
    sentences = re.split('(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        # Check if adding this sentence would exceed the limit
        if len((current_chunk + sentence).encode('utf-8')) > max_bytes:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip()
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def synthesize_long_audio(text: str, output_path: str) -> str:
    """Handle long text TTS by splitting into chunks and combining audio files."""
    # Initialize temp_files at the start to ensure it's always available
    temp_files = []
    try:
        # Verify FFmpeg is available
        try:
            # First try the standard Program Files location
            FFMPEG_PATH = r"C:\Program Files\ffmpeg\bin"
            if not os.path.exists(FFMPEG_PATH):
                # Fall back to the Downloads location
                FFMPEG_PATH = r"C:\Users\3d3n2\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin"
            
            if not os.path.exists(FFMPEG_PATH):
                raise RuntimeError("FFmpeg directory not found")
            
            # Check for required files
            required_files = ['ffmpeg.exe', 'ffprobe.exe']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(FFMPEG_PATH, f))]
            if missing_files:
                raise RuntimeError(f"Missing FFmpeg files: {', '.join(missing_files)}")
            
            # Add to PATH if not already there
            if FFMPEG_PATH not in os.environ['PATH']:
                os.environ['PATH'] = FFMPEG_PATH + os.pathsep + os.environ['PATH']
            
            # Explicitly set ffmpeg paths for pydub
            AudioSegment.converter = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
            AudioSegment.ffmpeg = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
            AudioSegment.ffprobe = os.path.join(FFMPEG_PATH, "ffprobe.exe")
            
            # Test FFmpeg by trying to create a silent audio segment
            try:
                # Create a 1ms silent audio segment to test FFmpeg
                AudioSegment.silent(duration=1)
                logger.debug("FFmpeg test successful")
            except Exception as e:
                raise RuntimeError(f"FFmpeg test failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"FFmpeg not properly configured: {str(e)}")
            raise RuntimeError(f"FFmpeg not properly configured. Please ensure FFmpeg is installed and in your system PATH. Error: {str(e)}")

        client = texttospeech.TextToSpeechClient()
        chunks = split_text_into_chunks(text)
        combined_audio = None

        logger.debug(f"Split text into {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            try:
                # Generate temporary file name
                temp_file = f"static/temp_{uuid.uuid4()}.mp3"
                temp_files.append(temp_file)
                
                synthesis_input = texttospeech.SynthesisInput(text=chunk)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US",
                    name="en-US-Studio-O",
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    speaking_rate=1.0,
                    pitch=0.0,
                    sample_rate_hertz=24000
                )
                
                logger.debug(f"Synthesizing chunk {i+1}/{len(chunks)}")
                response = client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                
                # Save temporary chunk
                with open(temp_file, "wb") as out:
                    out.write(response.audio_content)
                
                # Verify the temporary file was created
                if not os.path.exists(temp_file):
                    raise RuntimeError(f"Failed to create temporary audio file: {temp_file}")
                
                # Combine audio files using pydub
                if combined_audio is None:
                    combined_audio = AudioSegment.from_mp3(temp_file)
                else:
                    chunk_audio = AudioSegment.from_mp3(temp_file)
                    combined_audio += chunk_audio
                
                logger.debug(f"Successfully processed chunk {i+1}")

            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                raise RuntimeError(f"Error processing audio chunk {i+1}: {str(e)}")

        # Export final combined audio
        if combined_audio:
            logger.debug(f"Exporting final audio to {output_path}")
            combined_audio.export(output_path, format="mp3")
            
            if not os.path.exists(output_path):
                raise RuntimeError(f"Failed to create final audio file: {output_path}")
        else:
            raise RuntimeError("No audio was generated")
        
        return output_path
    except Exception as e:
        logger.error(f"Error in synthesize_long_audio: {str(e)}", exc_info=True)
        raise RuntimeError(f"Error generating audio: {str(e)}")
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file {temp_file}: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
