from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
import json

class Settings(BaseSettings):
    # API Security
    API_KEY: str | None = None
    RATE_LIMIT: str = "100/day"
    
    # CORS Settings
    FRONTEND_URL: str = "http://localhost:3000"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]
    
    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".txt"]
    
    # Google Cloud Settings
    GOOGLE_APPLICATION_CREDENTIALS: str = "./credentials/google-credentials.json"
    
    # OpenAI Settings
    OPENAI_API_KEY: str | None = None
    
    # Environment
    ENV: str = "development"
    DEBUG: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str):
            if field_name == "ALLOWED_ORIGINS":
                try:
                    # Try to parse as JSON first
                    return json.loads(raw_val)
                except json.JSONDecodeError:
                    # If not JSON, split by comma
                    return [origin.strip() for origin in raw_val.split(",")]
            return raw_val

@lru_cache()
def get_settings():
    return Settings()

# Security headers
security_headers = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "X-XSS-Protection": "1; mode=block",
    "Content-Security-Policy": "default-src 'self'",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
} 