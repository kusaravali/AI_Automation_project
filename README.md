# AI_Automation_project
# AI Automation

A FastAPI service that uses a Hugging Face sentiment model
(`distilbert-base-uncased-finetuned-sst-2-english`) to classify text and
suggest an automated action:

- NEGATIVE -> "Flag for customer support review"
- POSITIVE -> "Record as positive feedback"

## Installation
```
pip install -r requirements.txt
```

## Run the tests
```
python test_api.py
```

## Start the API
```
uvicorn main:app --reload
```
Then open http://127.0.0.1:8000/docs to try the `/analyse` endpoint.

## Example request
```
POST /analyse
{"text": "This product is terrible. I hate it."}
```

## Tech Stack
Python, FastAPI, Pydantic, PyTorch, Hugging Face Transformers
