# Ryzn Notes Backend

A FastAPI backend for document processing, summarization, and audio generation.

## Features

- Document text extraction (PDF, TXT)
- AI-powered document summarization
- Notes generation in markdown format
- Quiz generation from document content
- Text-to-speech conversion
- Audio transcription

## Requirements

- Python 3.8+
- FFmpeg
- Poppler
- Tesseract OCR
- Google Cloud credentials

## Setup

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/ryzn-notes.git
   cd ryzn-notes
   ```

2. Create and activate a virtual environment:

   ```
   cd src/app/backend
   python -m venv venv

   # Windows
   .\venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the `src/app/backend` directory with the following variables:

   ```
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_APPLICATION_CREDENTIALS=path_to_google_credentials.json
   ```

5. Install system dependencies:

   - Tesseract OCR (for Windows: https://github.com/UB-Mannheim/tesseract/wiki)
   - Poppler (for Windows: https://github.com/oschwartz10612/poppler-windows/releases/)
   - FFmpeg (for Windows: https://ffmpeg.org/download.html)

6. Update paths in `main.py` if necessary:
   - Tesseract path
   - Poppler path
   - FFmpeg path

## Deployment

1. Create required directories:

   ```
   mkdir -p uploads static cache credentials
   ```

2. Place your Google Cloud credentials in the credentials directory:

   ```
   cp path/to/your/google-credentials.json credentials/
   ```

3. Start the server:

   ```
   uvicorn src.app.backend.main:app --host 0.0.0.0 --port 8000
   ```

4. For production deployment, consider using a process manager like Gunicorn:
   ```
   pip install gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.app.backend.main:app
   ```

## API Endpoints

- `POST /api/summarize`: Summarize a document (PDF, TXT)
- `POST /api/response`: Generate a response to a user query about a document
- `POST /api/transcribe`: Transcribe an audio file

## Development

- To run tests: `pytest`
- To check code quality: `flake8`
- To format code: `black src/`

## License

[MIT](LICENSE)
