import multiprocessing
from .config import get_settings

settings = get_settings()

# Number of worker processes
workers = min(multiprocessing.cpu_count() * 2 + 1, 4)  # Limit max workers to 4

# Worker class
worker_class = 'uvicorn.workers.UvicornWorker'

# Maximum request body size (in bytes)
limit_request_line = 0  # Unlimited
limit_request_fields = 32768
limit_request_field_size = 0  # Unlimited

# Timeouts
timeout = 600  # 10 minutes
keepalive = 5

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'debug'

# Bind address
bind = '0.0.0.0:8000'

# Maximum request body size (matches FastAPI setting)
max_request_size = settings.MAX_UPLOAD_SIZE

# Worker settings
worker_tmp_dir = '/dev/shm'  # Use shared memory for temporary files
max_requests = 1000  # Restart workers after handling this many requests
max_requests_jitter = 50  # Add jitter to prevent all workers from restarting at once

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