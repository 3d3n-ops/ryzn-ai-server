# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import pytesseract
from pdf2image import convert_from_path
from langchain_groq import ChatGroq
from gtts import gTTS
from dotenv import load_dotenv
import whisper
from google.cloud import texttospeech
from pathlib import Path
import re
from pydub import AudioSegment
import time
import json
import hashlib
import threading
import subprocess

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables first
load_dotenv()

# Set up caching
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_key(file_content: str, output_type: str) -> str:
    """Generate a unique cache key based on file content and output type."""
    content_hash = hashlib.md5(file_content.encode()).hexdigest()
    return f"{content_hash}_{output_type}"

def get_cached_response(cache_key: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached response if it exists."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_data
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")
    return None

def cache_response(cache_key: str, response_data: Dict[str, Any]) -> None:
    """Cache the response data."""
    try:
        cache_file = CACHE_DIR / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(response_data, f)
        logger.debug(f"Cached response for key: {cache_key}")
    except Exception as e:
        logger.error(f"Error caching response: {str(e)}")

## These are neccesary paths and configurations for the app to run. Tesseract helps with extracting the text from the file, 
## Poppler helps with converting the PDF to images, and FFmpeg helps with combining the audio files.
## Google Cloud credentials are used to generate the audio.
## The paths are set up for Windows and Docker/Linux environments.

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CREDENTIALS_DIR = BASE_DIR / "credentials"
CREDENTIALS_FILE = CREDENTIALS_DIR / "google-credentials.json"

# Ensure directories exist
CREDENTIALS_DIR.mkdir(exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Check if we're running in Docker/Render
is_docker = os.environ.get('RENDER', 'false').lower() == 'true' or os.path.exists('/.dockerenv')

# Set paths for different operating systems
if os.name == 'nt' and not is_docker:  # Windows (local development)
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
    FFMPEG_PATH = r"C:\Users\3d3n2\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin"
    if not os.path.exists(FFMPEG_PATH):
        logger.error(f"FFmpeg not found at {FFMPEG_PATH}")
        raise RuntimeError(f"FFmpeg not found at {FFMPEG_PATH}. Please install FFmpeg and ensure it's in the correct location.")
    
    # Add FFmpeg to system PATH for pydub
    if FFMPEG_PATH not in os.environ['PATH']:
        os.environ['PATH'] += os.pathsep + FFMPEG_PATH
else:  # Linux/Docker/Render
    # In Docker, these should be installed in standard locations
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    POPPLER_PATH = '/usr/bin'
    FFMPEG_PATH = '/usr/bin'
    
    # Set the ffmpeg paths for pydub
    AudioSegment.converter = '/usr/bin/ffmpeg'
    AudioSegment.ffmpeg = '/usr/bin/ffmpeg'
    AudioSegment.ffprobe = '/usr/bin/ffprobe'
    
    logger.debug("Running in Docker/Linux environment")
    logger.debug(f"Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
    logger.debug(f"Poppler path: {POPPLER_PATH}")
    logger.debug(f"FFmpeg path: {FFMPEG_PATH}")

def verify_binaries():
    """Verify that required binaries are available"""
    try:
        logger.debug("Verifying binary installations...")
        subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
        logger.debug("Tesseract verified")
        
        subprocess.run(['pdftoppm', '-v'], capture_output=True, check=True)
        logger.debug("Poppler verified")
        
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        logger.debug("FFmpeg verified")
        
        logger.debug("All required binaries verified successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Binary verification failed: {str(e)}")
        raise RuntimeError(f"Required binary verification failed: {str(e)}")
    except Exception as e:
        logger.error(f"Binary verification failed: {str(e)}")
        raise RuntimeError(f"Required binary verification failed: {str(e)}")


# Verify Google Cloud credentials
google_creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if not google_creds_path:
    logger.error("GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
    raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set in .env file. Please add it to your .env file.")

# Convert relative path to absolute if necessary
if not os.path.isabs(google_creds_path):
    google_creds_path = str(BASE_DIR / google_creds_path)

# Only check for credentials file in development mode
if os.environ.get('RENDER') != 'true' and not os.path.exists(google_creds_path):
    logger.error(f"Google Cloud credentials file not found at: {google_creds_path}")
    logger.error(f"Please place your google-credentials.json file in: {CREDENTIALS_DIR}")
    raise RuntimeError(f"Google Cloud credentials file not found. Please place your google-credentials.json file in: {CREDENTIALS_DIR}")

# Set the environment variable with the absolute path
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds_path
logger.debug(f"Using Google Cloud credentials from: {google_creds_path}")

app = FastAPI()

# Get the list of allowed origins from environment or use default
allowed_origins = ["http://localhost:3000", "https://rzn-ai.vercel.app", "https://ryzn-ai-server.onrender.com"]

# CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for serving audio files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize LLM with Groq
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

llm = ChatGroq(
    model_name="mixtral-8x7b-32768",  # Using Mixtral model from Groq
    temperature=0.7,
    api_key=groq_api_key,
    streaming=False
)

# Add debug logging for API key
logger.debug(f"Using Groq API key: {groq_api_key[:8]}...")  # Only log first 8 chars for security

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
    quiz: Optional[Dict[str, Any]] = None

# Add a simple rate limiter
class RateLimiter:
    def __init__(self, max_calls=5, time_period=60):
        self.max_calls = max_calls
        self.time_period = time_period
        self.calls = []
        self.lock = False
    
    def can_call(self):
        current_time = time.time()
        # Remove old calls
        self.calls = [call_time for call_time in self.calls if current_time - call_time < self.time_period]
        
        # Check if we can make a new call
        if len(self.calls) < self.max_calls and not self.lock:
            self.calls.append(current_time)
            return True
        return False
    
    def add_cooldown(self, seconds=10):
        """Add a cooldown period where no calls are allowed"""
        self.lock = True
        threading.Timer(seconds, self._release_lock).start()
    
    def _release_lock(self):
        self.lock = False

# Create rate limiter for Groq API
groq_limiter = RateLimiter(max_calls=12, time_period=86400)  # 182 calls per day

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

# Call verify_binaries during startup to ensure all required tools are available
@app.on_event("startup")
async def startup_event():
    try:
        verify_binaries()
    except Exception as e:
        logger.error(f"Failed to verify required binaries during startup: {str(e)}")
        # Log but don't crash, as the binary might be available in a different way

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
        
        # Check cache first
        cache_key = get_cache_key(text, output_type)
        cached_response = get_cached_response(cache_key)
        if cached_response:
            logger.debug("Using cached response")
            return SummaryResponse(**cached_response)
        
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
        
        # Create response object
        response_data = {
            "summary": summary,
            "audio_url": audio_url,
            "notes": notes,
            "tts_audio_url": tts_audio_url,
            "quiz": quiz
        }
        
        # Cache the response
        cache_response(cache_key, response_data)
        
        return SummaryResponse(**response_data)
    
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

def chunk_text(text: str, max_chunk_size: int = 2000) -> List[str]:
    """Split text into smaller chunks while preserving sentence boundaries."""
    sentences = text.split('.')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence = sentence.strip() + '.'
        sentence_size = len(sentence)
        
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def generate_summary(text):
    try:
        chunks = chunk_text(text)
        summaries = []
        
        # Check cache for summaries
        cache_key = get_cache_key(text, "summary")
        cached_summary = get_cached_response(cache_key)
        if cached_summary:
            logger.debug("Using cached summary")
            return cached_summary.get("summary", "")
        
        for chunk in chunks:
            try:
                # Check if we're within rate limits
                if not groq_limiter.can_call():
                    logger.warning("Rate limit reached, waiting before making API call")
                    time.sleep(20)  # Wait 20 seconds when rate limited
                
                prompt = f"""Please provide a concise summary of the following text:

                {chunk}

                Focus on the main points and key takeaways. Be direct and concise."""
                
                logger.debug(f"Generating summary for chunk of size {len(chunk)}")
                response = llm.invoke(prompt)
                
                # Handle Groq's response format
                if hasattr(response, 'content'):
                    summaries.append(response.content)
                else:
                    summaries.append(str(response))
                    
                # Prevent rate limit issues by adding delay between calls
                time.sleep(3)
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error processing chunk: {error_msg}")
                
                # Check for rate limit errors
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    logger.warning("Rate limit hit, adding cooldown")
                    groq_limiter.add_cooldown(30)  # 30 second cooldown
                    time.sleep(30)
                
                raise HTTPException(status_code=500, detail=f"Error generating summary: {error_msg}")
        
        # Combine summaries
        if len(summaries) > 1:
            try:
                # Check if we're within rate limits
                if not groq_limiter.can_call():
                    logger.warning("Rate limit reached for final summary, using first chunk summary")
                    final_summary = summaries[0]
                else:
                    final_prompt = f"""Please provide a concise final summary combining these summaries:

                    {' '.join(summaries)}

                    Focus on the main points and key takeaways. Be direct and concise."""
                    
                    final_response = llm.invoke(final_prompt)
                    
                    # Handle Groq's response format
                    if hasattr(final_response, 'content'):
                        final_summary = final_response.content
                    else:
                        final_summary = str(final_response)
                
                # Cache the result
                cache_response(cache_key, {"summary": final_summary})
                return final_summary
                    
            except Exception as e:
                logger.error(f"Error combining summaries: {str(e)}")
                # If combining fails, return the first summary
                return summaries[0]
        else:
            # Cache the result
            cache_response(cache_key, {"summary": summaries[0]})
            return summaries[0]
            
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

def generate_quiz(text):
    # Check cache first for this specific text and quiz
    cache_key = get_cache_key(text, "quiz")
    cached_quiz = get_cached_response(cache_key)
    if cached_quiz:
        logger.debug(f"Using cached quiz")
        return cached_quiz.get("quiz")
    
    # Truncate text to avoid token limits
    truncated_text = text[:4000]
    prompt = f"""With the context of this text: {truncated_text}, generate a quiz with 5-10 questions.
    Format your response as a JSON object with the following structure:
    {{
        "title": "Quiz Title",
        "description": "Brief description of the quiz",
        "questions": [
            {{
                "id": "q1",
                "question": "Question text",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correctAnswer": "Correct option text",
                "explanation": "Explanation of why this is correct"
            }}
        ]
    }}
    Make sure to include 5-10 questions and format the response as valid JSON."""
    
    try:
        logger.debug("Generating quiz")
        
        # Check if we're within rate limits
        if not groq_limiter.can_call():
            logger.warning("Rate limit reached, waiting before generating quiz")
            time.sleep(20)
        
        response = llm.invoke(prompt)
        
        # Extract the JSON string from the response
        content = response.content
        # Find the JSON object in the response
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No valid JSON found in response")
            
        json_str = content[json_start:json_end]
        
        # Parse the JSON string into a Python dictionary
        quiz_data = json.loads(json_str)
        
        # Validate the structure
        required_fields = ['title', 'description', 'questions']
        for field in required_fields:
            if field not in quiz_data:
                raise ValueError(f"Missing required field: {field}")
                
        if not isinstance(quiz_data['questions'], list):
            raise ValueError("Questions must be a list")
            
        for question in quiz_data['questions']:
            required_question_fields = ['id', 'question', 'options', 'correctAnswer', 'explanation']
            for field in required_question_fields:
                if field not in question:
                    raise ValueError(f"Missing required field in question: {field}")
        
        # Cache the quiz data
        cache_response(cache_key, {"quiz": quiz_data})
        
        return quiz_data
        
    except Exception as e:
        logger.error(f"Error generating quiz: {str(e)}")
        
        # Check for rate limit errors
        error_msg = str(e)
        if "429" in error_msg or "Too Many Requests" in error_msg:
            logger.warning("Rate limit hit during quiz generation, adding cooldown")
            groq_limiter.add_cooldown(30)
            
            # Try to create a simple fallback quiz to avoid complete failure
            fallback_quiz = {
                "title": "Fallback Quiz",
                "description": "Sorry, we couldn't generate a detailed quiz due to API limits. Here's a simple quiz instead.",
                "questions": [
                    {
                        "id": "q1",
                        "question": "What's the primary topic of the uploaded document?",
                        "options": ["Please review the document", "Option B", "Option C", "Option D"],
                        "correctAnswer": "Please review the document",
                        "explanation": "This is a fallback quiz due to API limitations."
                    }
                ]
            }
            return fallback_quiz
            
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
        # Verify FFmpeg is available - different approach based on environment
        if os.name == 'nt' and not is_docker:  # Windows local development
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
            except Exception as e:
                raise RuntimeError(f"FFmpeg not properly configured on Windows: {str(e)}")
        else:  # Docker/Render environment
            # In Docker, ffmpeg should be available in standard locations
            AudioSegment.converter = '/usr/bin/ffmpeg'
            AudioSegment.ffmpeg = '/usr/bin/ffmpeg'
            AudioSegment.ffprobe = '/usr/bin/ffprobe'
            
            logger.debug("Using Docker/Linux FFmpeg configuration")
            
        # Test FFmpeg by trying to create a silent audio segment
        try:
            # Create a 1ms silent audio segment to test FFmpeg
            AudioSegment.silent(duration=1)
            logger.debug("FFmpeg test successful")
        except Exception as e:
            raise RuntimeError(f"FFmpeg test failed: {str(e)}")

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
    import os
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"  # Important for Render
    uvicorn.run(app, host=host, port=port)

