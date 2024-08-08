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

# Get the tuned model
tuned_model_name = "tunedModels/brightspend-ai-training-11mks2kex5wq"
tuned_model = genai.get_tuned_model(tuned_model_name)

# Print the details of the tuned model
print(f"Tuned Model Name: {tuned_model.name}")

# set the tuned model
model = genai.GenerativeModel(model_name=tuned_model.name)

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
    input_text = data.get("input", "")

    if not input_text:
        return jsonify({"error": "Input text is required"}), 400

    try:
        response = model.generate_content(input_text)
        return jsonify({"response": response.text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Simple root endpoint
@app.route('/')
def root():
    return jsonify({"message": "Welcome to the Generative AI API!"})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)