# Use Ray's ML-ready GPU image as base
FROM rayproject/ray-ml:latest-gpu

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY app.py .

# Expose the port that Ray Serve will use
EXPOSE 8000