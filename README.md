# Mental Health Sentiment App

## Problem Description

This project classifies mental health-related text into supportive categories using a fine-tuned sentiment analysis model. The goal is to help users and professionals quickly assess the emotional tone of mental health expressions, enabling better support and triage.

The input is a single text string (e.g., a journal entry, message, or post). The output is a message containing one of the following labels:
- Anxiety
- Bipolar
- Depression
- Normal
- Personality disorder
- Stress
- Suicidal

## Model & Inference

Model: Fine-tuned Hugging Face transformer for mental health sentiment classification.

Framework: PyTorch + Transformers

Inference: CPU-only, optimized for lightweight deployment

## App Workflow

Flask-based web app with a single /predict endpoint

Accepts POST requests with JSON payload: {"text": "..."}

Returns prediction in the form of a message

## Reproducibility

Model weights and model are saved in app to reduce memory consumption in cloud service

Production dependencies listed in `requirements.txt` and development in `requirements_dev.txt`

App runs locally or in containerized environments

## Deployment

Cloud Platform: Google Cloud Run

Containerization: Docker

Public Endpoint: [https://mentalhealth-sentiment-app-646188564826.asia-east1.run.app/](https://mentalhealth-sentiment-app-646188564826.asia-east1.run.app/)

## Dependency & Environment Management

Use a virtual environment for consistent results:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_dev.txt
```

## Containerization

Dockerfile included to build and run the app:

```
docker build -t mentalhealth-sentiment-app .
docker run -p 8080:8080 mentalhealth-sentiment-app
```

## Future Improvements

Expand label set for more nuanced emotional categories
