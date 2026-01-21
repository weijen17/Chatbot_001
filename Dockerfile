FROM python:3.12-slim

WORKDIR /app

# Set environment variables (adds metadata)
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

RUN mkdir -p logs

#############################################
# Declare a build argument
ARG EMBED_MODEL

# Set it as an environment variable inside the container
ENV EMBED_MODEL=${EMBED_MODEL}

# Download the model during build
RUN python - <<EOF
import os
from sentence_transformers import SentenceTransformer

model_name = os.environ.get("EMBED_MODEL")
SentenceTransformer(model_name)
EOF

#############################################

# Dummy port, even if not actually used
EXPOSE 5000

# May be needed to prevent error on docker-compose up
# VOLUME ["/app/outputs", "/app/src"]

# Run the application
#CMD ["python", "main.py"]

# Run the application and perform evaluation
 CMD sh -c "streamlit run main.py"
