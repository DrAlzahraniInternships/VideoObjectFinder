FROM python:3.11-slim

# ffmpeg for frame extraction; ca-certs for HTTPS model downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY app.py ./app.py
COPY python ./python
COPY templates ./templates
RUN mkdir -p /app/jobs

# Install PyTorch CPU wheels first, then the rest (keeps image smaller/quicker)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.8.0 torchvision==0.23.0 \
 && pip install --no-cache-dir -r requirements.txt

EXPOSE 3000
ENV PORT=3000
CMD ["python","-u","/app/app.py"]
