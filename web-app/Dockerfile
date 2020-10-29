FROM pytorch/pytorch
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT [ "python3" ]
CMD [ "main.py" ]
