# tests/deployment/test_deployment.py
import os
import json
import pytest
import requests

# The deployed endpoint URL
DEPLOYED_ENDPOINT = os.environ.get("API_ENDPOINT")

# If your API requires an API key, set it here or load from environment variables
API_KEY = os.getenv("REPLICATE_API_TOKEN")

@pytest.mark.deployment
def test_deployed_endpoint_basic():
    """
    Test the deployed endpoint with a basic request.
    """
    payload = {
        "prompt": "Describe the image.",
        "image_urls": "https://github.com/JimothyJohn/cerebro/blob/master/data/images/zidane.jpg?raw=true",
        "max_tokens": 50,
        "temperature": 0.7,
        "do_sample": True
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(
        DEPLOYED_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=30  # Set a timeout to avoid hanging indefinitely
    )

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

    body = response.json()

    # Assert the expected keys are present
    assert "choices" in body, "'choices' not in response body"
    assert len(body["choices"]) > 0, "No choices returned in response"
    assert "text" in body["choices"][0], "'text' not in choices[0]"

@pytest.mark.deployment
def test_deployed_endpoint_count_people():
    """
    Test the deployed endpoint to count the number of people in the image.
    """
    prompt = (
        "Tell me how many people are in this image. Your answer should be just a single number, for example '2'. "
        "When I ask you how many people are in the image you must only output a number with no additional text. "
        "For example if I asked you how many people were in an image and there were 2 people you would only respond with '2'."
    )

    payload = {
        "prompt": prompt,
        "image_urls": "https://github.com/JimothyJohn/cerebro/blob/master/data/images/zidane.jpg?raw=true",
        "max_tokens": 10,
        "temperature": 0.0,
        "do_sample": False
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(
        DEPLOYED_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=30
    )

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

    body = response.json()
    generated_text = body["choices"][0]["text"].strip()

    # Try to convert the response to an integer
    try:
        result = int(generated_text)
    except ValueError:
        pytest.fail(f"Expected an integer response, but got: '{generated_text}'")

    assert result >= 0, "The number of people cannot be negative."

@pytest.mark.deployment
def test_deployed_endpoint_count_subjects():
    """
    Test the deployed endpoint to count the number of subjects in the image.
    """
    prompt = (
        "Count the number of subjects in the image. There will be an obvious background and foreground, "
        "and you will tell me how many objects you see in the foreground. Your answer should be just a single number, "
        "for example '2'. When I ask you how many people are in the image you must only output a number with no additional text. "
        "For example if I asked you how many people were in an image and there were 2 people you would only respond with '2'."
    )

    payload = {
        "prompt": prompt,
        "image_urls": "https://github.com/JimothyJohn/cerebro/blob/master/data/images/zidane.jpg?raw=true",
        "max_tokens": 10,
        "temperature": 0.0,
        "do_sample": False
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(
        DEPLOYED_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=30
    )

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

    body = response.json()
    generated_text = body["choices"][0]["text"].strip()

    # Try to convert the response to an integer
    try:
        result = int(generated_text)
    except ValueError:
        pytest.fail(f"Expected an integer response, but got: '{generated_text}'")

    assert result >= 0, "The number of subjects cannot be negative."
