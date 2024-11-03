import io
from google.cloud import vision
import os

# Set up Google Cloud credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/service_account.json"  # Replace with your JSON file path

def analyze_image(image_path):
    """Recognize image labels and return the results"""
    # Initialize Vision API client
    client = vision.ImageAnnotatorClient()

    # Read image file
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    # Create image object for the Vision API
    image = vision.Image(content=content)

    # Request label detection
    response = client.label_detection(image=image)
    labels = response.label_annotations

    # Display results
    results = []
    for label in labels:
        results.append((label.description, label.score))
        print(f"Label: {label.description}, Score: {label.score}")

    # Error handling
    if response.error.message:
        raise Exception(f"{response.error.message}")

    return results

# Example usage
image_path = "example.jpg"  # Path to the image file taken on a smartphone
result = analyze_image(image_path)
print("Recognition Results:", result)
