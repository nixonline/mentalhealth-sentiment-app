FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY predict.py train.py .
COPY templates/ ./templates
COPY static/ ./static

COPY model/ ./model/

COPY model_state.pt .
COPY binary_model_state.pt .

EXPOSE 5000
CMD ["python", "predict.py"]