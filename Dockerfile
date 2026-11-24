FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Define a pasta de cache do Hugging Face para o disco principal do container
ENV HF_HOME=/app/cache_huggingface
ENV TORCH_HOME=/app/cache_torch

RUN apt-get update && apt-get install -y ffmpeg libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
