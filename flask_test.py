from dotenv import load_dotenv
from flask import Flask, request, jsonify
from marshmallow import Schema, fields, ValidationError
import google.generativeai as genai
import os
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# Load environment variables
load_dotenv()

API_KEY = os.getenv('GEMINI_API_KEY')

# Configure the API key for Google Generative AI
genai.configure(api_key=API_KEY)

# Define the generation configuration
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "max_output_tokens": 8192,
    # Remove response_mime_type if it's not a supported field
}

# Initialize the generative model
"""
model = genai.GenerativeModel(
    model_name="tunedModels/brightspend-ai-training-ilpn6zzcubfi",
    generation_config=generation_config,
)
"""

model = genai.GenerativeModel('gemini-pro')

# Define the request schema using Marshmallow
class GenerationRequestSchema(Schema):
    input_text = fields.Str(required=True)

# Initialize Flask app
app = Flask(__name__)

# Endpoint for generating content
@app.route('/generate/', methods=['POST'])
def generate_content():
    schema = GenerationRequestSchema()
    try:
        # Validate and deserialize input data
        data = schema.load(request.json)
    except ValidationError as err:
        return jsonify(err.messages), 400

    input_text = data['input_text']
    try:
        # Generate content using the input text
        response = model.generate_content([f"input: {input_text}", "output: "])
        return jsonify({"output": response.text})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# Simple root endpoint
@app.route('/')
def root():
    return jsonify({"message": "Welcome to the Generative AI API!"})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
