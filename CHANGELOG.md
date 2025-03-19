# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-03-19

### Added

- Initial FastAPI backend setup
- Document processing functionality:
  - PDF text extraction using Tesseract OCR and pdf2image
  - Text file parsing
  - Document summarization using Groq LLM
  - Notes generation in markdown format
  - Quiz generation from document content
- Audio features:
  - Text-to-speech conversion using Google Cloud TTS
  - Audio file generation from summaries
  - Long audio synthesis with text chunking
  - Audio transcription using Whisper
- Caching system for API responses
- Rate limiting for external API calls
- CORS middleware for frontend integration
- Error handling and logging
- Static file serving for generated audio files
- Environment variable configuration with dotenv
- Virtual environment setup

### Fixed

- Environment configuration for Windows compatibility
- FFmpeg, Poppler, and Tesseract path configuration
- Import resolution issues
- API path organization

### Security

- Added rate limiting to prevent API abuse
- Implemented proper error handling to prevent information leakage
- Added file validation for uploads

## [Unreleased]

- Database integration for persistent storage
- User authentication and authorization
- Additional output formats
- Improved error handling
- Performance optimizations for large documents
- Enhanced caching mechanism
