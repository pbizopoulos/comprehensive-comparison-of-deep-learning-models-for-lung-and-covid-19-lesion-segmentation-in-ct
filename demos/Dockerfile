FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt
COPY . .
