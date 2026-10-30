"""
AI Sentiment Automation API
FastAPI app that analyses text sentiment and suggests an action.
"""

from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints
from transformers import pipeline

# Load the model once when the app starts.
analyser = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt",
    device=-1,  # Use CPU.
)

app = FastAPI(title="AI Sentiment Automation")

# Reject empty, whitespace-only, and excessively long input.
Text = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=5000,
    ),
]


class SentimentRequest(BaseModel):
    text: Text


@app.get("/")
def home():
    return {"message": "Sentiment API is ready"}


@app.post("/analyse")
def analyse_sentiment(request: SentimentRequest):
    prediction = analyser(
        request.text,
        truncation=True,
        max_length=512,
    )[0]

    sentiment = prediction["label"]

    # Simple automation rule based on the AI prediction.
    action = (
        "Flag for customer support review"
        if sentiment == "NEGATIVE"
        else "Record as positive feedback"
    )

    return {
        "text": request.text,
        "sentiment": sentiment,
        "confidence": round(float(prediction["score"]), 4),
        "suggested_action": action,
    }
