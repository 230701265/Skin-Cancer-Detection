FROM --platform=linux/arm64 python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create model directory
RUN mkdir -p model

# Expose the port the app will run on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"] 