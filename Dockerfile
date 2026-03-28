# Use a more complete Python base image - Cache bust v2
FROM python:3.9

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install OS dependencies required by OpenCV and skimage-like libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx0 \
    libxrender1 \
    libx11-6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_minimal.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY templates/ templates/
COPY static/ static/

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
