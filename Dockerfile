# Stage 1: Build stage
FROM python:3.9-slim-buster 

RUN apt-get update -y && \
    apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    git \
    # cleanup
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists

COPY requirements.txt .

RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir --requirement requirements.txt

# Stage 2: Runtime stage

LABEL maintainer="team-erc"

ENV WORKERS_PER_CORE=4 
ENV MAX_WORKERS=24
ENV LOG_LEVEL="warning"
ENV TIMEOUT="200"

WORKDIR /app

COPY . .

EXPOSE 80

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
