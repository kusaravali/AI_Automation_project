"""
Tests for the AI Sentiment Automation API.
Run with:  python test_api.py
"""

from fastapi.testclient import TestClient

from main import app

with TestClient(app) as client:
    # Test 1: API health.
    response = client.get("/")
    assert response.status_code == 200

    # Test 2: Positive sentiment and its automation rule.
    response = client.post(
        "/analyse",
        json={"text": "I love this product! It is excellent."},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["sentiment"] == "POSITIVE"
    assert 0 <= result["confidence"] <= 1
    assert result["suggested_action"] == "Record as positive feedback"
    print("Positive result:", result)

    # Test 3: Negative sentiment and its automation rule.
    response = client.post(
        "/analyse",
        json={"text": "This product is terrible. I hate it."},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["sentiment"] == "NEGATIVE"
    assert result["suggested_action"] == "Flag for customer support review"
    print("Negative result:", result)

    # Test 4: Empty or whitespace-only input.
    for text in ["", " "]:
        response = client.post("/analyse", json={"text": text})
        assert response.status_code == 422

    # Test 5: Missing required field.
    response = client.post("/analyse", json={})
    assert response.status_code == 422

print("\nAll tests passed!")
