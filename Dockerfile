# 1. Base Image: Lightweight Python 3.12 on Debian Linux
FROM python:3.12-slim

# 2. Prevent Python from writing .pyc files and enable live stdout logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set working directory inside the container
WORKDIR /app

# 4. Install system C++ build dependencies required by ChromaDB
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your application source code
COPY . .

# 7. Expose the FastAPI gateway port
EXPOSE 8000

# 8. Boot up Uvicorn server bound to all network interfaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
