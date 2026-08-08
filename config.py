#Load environment variables and securely initialize the OpenAI client using the API key from .env.
#Add optional Hugging Face API client configuration for future AI model integration.

import os
from pathlib import Path
from dotenv import load_dotenv

# OpenAI
from openai import OpenAI

# Find project root: parent directory of src/
PROJECT_ROOT = Path(__file__).resolve().parent

#Load .env from PROJECT_ROOT
load_dotenv(PROJECT_ROOT/".env")
ENV_FILE = PROJECT_ROOT/".env"

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        f"OPENAI_API_KEY not found. Please check .env file at: {ENV_FILE}"
    )

client = OpenAI(api_key=api_key)

#Hugging face 
# from huggingface_hub import InferenceClient

# load_dotenv(PROJECT_ROOT / ".env")

# hf_api_key = os.getenv("HF_API_KEY")

# if not hf_api_key:
#     raise RuntimeError(
#         f"HF_API_KEY not found. Please check: {PROJECT_ROOT / '.env'}"
#     )

# client = InferenceClient(
#     api_key=hf_api_key
# )
