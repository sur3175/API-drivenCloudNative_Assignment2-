import os
from pathlib import Path
from dotenv import load_dotenv

# OpenAI
from openai import OpenAI
from huggingface_hub import InferenceClient

# Find project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Directory to store generated images
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

#Load .env from PROJECT_ROOT
load_dotenv(PROJECT_ROOT/".env")
ENV_FILE = PROJECT_ROOT/".env"

def create_client(provider:str):
    provider = provider.lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"OPENAI_API_KEY not found. Please check .env file at: {ENV_FILE}"
            )
        return OpenAI(api_key=api_key)
    elif provider == "hf":
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"HF_API_KEY not found. Please check .env file at: {ENV_FILE}"
            )
        return InferenceClient(api_key=api_key)
    else:
        raise ValueError(
            f"Unsupported provider '{provider}'. "
            "Use 'openai' or 'hf'."
        )
