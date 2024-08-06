from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import google.generativeai as genai
import os
from pydantic import BaseModel

# load environmental variables
load_dotenv()

API_KEY = os.getenv('GEMINI_API_KEY')

# Define the generation configuration
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the generative model
model = genai.GenerativeModel(
    model_name="tunedModels/brightspend-ai-training-ilpn6zzcubfi",
    generation_config=generation_config,
)

# Define the request model
class GenerationRequest(BaseModel):
    input_text: str

# Create the FastAPI app
app = FastAPI()

# Define the endpoint for generating content
@app.post("/generate/")
async def generate_content(request: GenerationRequest):
    try:
        # Generate content using the input text
        response = model.generate_content([f"input: {request.input_text}", "output: "])
        return {"output": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example of a simple endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Generative AI API!"}