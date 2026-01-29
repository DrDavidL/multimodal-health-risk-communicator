FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY demo.py .
COPY examples/ examples/

# Create non-root user for security
RUN useradd -m -u 1000 user
USER user

# Expose Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "demo.py"]
