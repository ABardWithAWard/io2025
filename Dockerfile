# Build stage for Node.js
FROM node:18 AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
COPY frontend/.env* ./
ENV HTTPS=true
RUN npm run build

FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Copy environment files if they were not copied
COPY ocr/.env* ./ocr/

# Copy built frontend from the builder stage
COPY --from=frontend-builder /app/frontend/build /app/frontend/build

# Generate SSL certificates if they don't exist
RUN if [ ! -f cert.pem ] || [ ! -f key.pem ]; then \
    openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365 -subj "/CN=localhost"; \
    fi

RUN mkdir -p /app/ocr/application/model/modelMatthew/uploaded_files && \
    chmod 755 /app/ocr/application/model/modelMatthew/uploaded_files

RUN pip install paddle

RUN python ocr/manage.py collectstatic --noinput

EXPOSE 8000

# Start the server
CMD ["python", "ocr/manage.py", "runserver_plus", "--cert-file", "cert.pem", "--key-file", "key.pem", "0.0.0.0:8000"] 