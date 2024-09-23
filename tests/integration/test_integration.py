# tests/integration/test_integration.py
import json
import os
import pytest
from cerebro.app import lambda_handler

# Ensure REPLICATE_API_TOKEN is set
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

@pytest.mark.integration
def test_integration_lambda_handler_basic():
    """
    Basic integration test to check if the Lambda function returns a valid response.
    """
    # Set up event data simulating a real API Gateway request
    event = {
        "httpMethod": "POST",
        "body": json.dumps({
            "prompt": "Describe the image.",
            "image_urls": "https://github.com/JimothyJohn/cerebro/blob/master/data/images/zidane.jpg?raw=true",
            "max_tokens": 50,
            "temperature": 0.7,
            "do_sample": True
        }),
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}"
        },
        "isBase64Encoded": False,
        "path": "/v1/completions",
        "requestContext": {
            "httpMethod": "POST",
            "resourcePath": "/v1/completions"
        }
    }

    # Invoke the Lambda function
    response = lambda_handler(event, None)

    # Assert the response status code
    assert response["statusCode"] == 200

    # Parse the response body
    body = json.loads(response["body"])

    # Assert the expected keys are present
    assert "choices" in body
    assert len(body["choices"]) > 0
    assert "text" in body["choices"][0]

@pytest.mark.integration
def test_integration_lambda_handler_count_people():
    """
    Test to ensure the model returns an integer when asked to count the number of people in the image.
    """
    prompt = (
        "Tell me how many people are in this image. Your answer should be just a single number, for example '2'. "
        "When I ask you how many people are in the image you must only output a number with no additional text. "
        "For example if I asked you how many people were in an image and there were 2 people you would only respond with '2'."
    )

    # Set up event data
    event = {
        "httpMethod": "POST",
        "body": json.dumps({
            "prompt": prompt,
            "image_urls": "https://github.com/JimothyJohn/cerebro/blob/master/data/images/zidane.jpg?raw=true",
            "max_tokens": 10,
            "temperature": 0.0,
            "do_sample": False
        }),
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}"
        },
        "isBase64Encoded": False,
        "path": "/v1/completions",
        "requestContext": {
            "httpMethod": "POST",
            "resourcePath": "/v1/completions"
        }
    }

    # Invoke the Lambda function
    response = lambda_handler(event, None)

    # Assert the response status code
    assert response["statusCode"] == 200

    # Parse the response body
    body = json.loads(response["body"])

    # Extract the generated text
    generated_text = body["choices"][0]["text"].strip()

    # Try to convert the response to an integer
    try:
        result = int(generated_text)
    except ValueError:
        pytest.fail(f"Expected an integer response, but got: '{generated_text}'")

    # Optionally, assert that the number is within an expected range
    assert result == 2, "The number of people cannot be negative."

@pytest.mark.integration
def test_integration_lambda_handler_count_subjects():
    """
    Test to ensure the model returns an integer when asked to count the number of subjects in the image.
    """
    prompt = (
        "Count the number of subjects in the image. There will be an obvious background and foreground, "
        "and you will tell me how many objects you see in the foreground. Your answer should be just a single number, "
        "for example '2'. When I ask you how many people are in the image you must only output a number with no additional text. "
        "For example if I asked you how many people were in an image and there were 2 people you would only respond with '2'."
    )

    # Set up event data
    event = {
        "httpMethod": "POST",
        "body": json.dumps({
            "prompt": prompt,
            "image_urls": "https://github.com/JimothyJohn/cerebro/blob/master/data/images/zidane.jpg?raw=true",
            "max_tokens": 10,
            "temperature": 0.0,
            "do_sample": False
        }),
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}"
        },
        "isBase64Encoded": False,
        "path": "/v1/completions",
        "requestContext": {
            "httpMethod": "POST",
            "resourcePath": "/v1/completions"
        }
    }

    # Invoke the Lambda function
    response = lambda_handler(event, None)

    # Assert the response status code
    assert response["statusCode"] == 200

    # Parse the response body
    body = json.loads(response["body"])

    # Extract the generated text
    generated_text = body["choices"][0]["text"].strip()

    # Try to convert the response to an integer
    try:
        result = int(generated_text)
    except ValueError:
        pytest.fail(f"Expected an integer response, but got: '{generated_text}'")

    # Optionally, assert that the number is within an expected range
    assert result == 2, "The number of subjects cannot be negative."
