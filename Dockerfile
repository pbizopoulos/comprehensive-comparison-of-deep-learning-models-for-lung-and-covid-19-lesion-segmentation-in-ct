FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt
