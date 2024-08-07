from dotenv import load_dotenv
from flask import Flask, request, jsonify
import google.generativeai as genai
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from load_creds import load_creds
import os
from marshmallow import Schema, fields, ValidationError

# Load environment variables
load_dotenv()

API_KEY = os.getenv('GEMINI_API_KEY')

# Configure the API key for Google Generative AI
genai.configure(api_key=API_KEY)

creds = load_creds()

# Configure Generative AI with credentials
genai.configure(credentials=creds)

# Define the generation configuration
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "max_output_tokens": 8192,
    # Remove response_mime_type if it's not a supported field
}

# Get the tuned model
tuned_model_name = "tunedModels/brightspend-ai-training-ilpn6zzcubfi"
model = genai.get_tuned_model(tuned_model_name)

"""
# Using gemini_pro as an example
model = genai.GenerativeModel('gemini-pro')
"""

# Initialize Flask app
app = Flask(__name__)

@app.route('/list_models', methods=['GET'])
def list_models():
    try:
        models = genai.list_models()
        model_names = [model.name for model in models]
        return jsonify({"models": model_names})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint for generating content
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    question = data.get("input", "")

    # Send the user's question along with the instruction to the chat model
    response = genai.generate_text(
        model=tuned_model_name,
        prompt=f"input: {question}\noutput: ",
        **generation_config
    )

    if not response:
        return jsonify({"error": "Input text is required"}), 400

    return jsonify({"response": response.text})

# Simple root endpoint
@app.route('/')
def root():
    return jsonify({"message": "Welcome to the Generative AI API!"})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)