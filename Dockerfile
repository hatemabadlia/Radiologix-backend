FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install "numpy==1.26.4" && \
    pip install torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install "ultralytics==8.3.0" && \
    pip install fastapi uvicorn python-multipart pillow pyyaml psutil && \
    pip install "opencv-python-headless==4.8.1.78" && \
    pip install "numpy==1.26.4" --force-reinstall

COPY main.py .
COPY best.pt .

EXPOSE 8000
CMD uvicorn main:app --host 0.0.0.0 --port $PORT