import multiprocessing
from .config import get_settings

settings = get_settings()

# Number of worker processes
workers = multiprocessing.cpu_count() * 2 + 1

# Worker class
worker_class = 'uvicorn.workers.UvicornWorker'

# Maximum request body size (in bytes)
limit_request_line = 0  # Unlimited
limit_request_fields = 32768
limit_request_field_size = 0  # Unlimited

# Timeouts
timeout = 300  # 5 minutes
keepalive = 5

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'debug'

# Bind address
bind = '0.0.0.0:8000'

# Maximum request body size (matches FastAPI setting)
max_request_size = settings.MAX_UPLOAD_SIZE 