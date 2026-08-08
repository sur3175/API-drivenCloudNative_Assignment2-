# We are using Streamlit which is python framework that lets us quickly turn 
# python code—especially ML/AI models—into an interactive web application.
import sys
import streamlit as sl
from src.text_generation import generate_text

from src.image_generation import generate_image
from src.image_generation import save_image

from config import create_client

# Validate and process the command-line provider argument.
# The provider must be either "openai" or "hf"; based on the selected
# provider, create_client() initializes the corresponding API client.
if len(sys.argv) < 2:
    raise RuntimeError("Usage: streamlit run app.py -- [openai|hf]")

provider = sys.argv[1].lower()

if provider not in ("openai", "hf"):
    raise RuntimeError("Provider must be 'openai' or 'hf'")

client = create_client(provider)
sl.sidebar.write(f"Provider: {provider.upper()}")

# Application title
sl.title("AI Application - Education")

#Text Generator 
sl.header("Text Generator")

prompt = sl.text_area(
    "Enter your prompt for text generation:",placeholder="" 
)

if sl.button("Generate Text"):
    if not prompt.strip():
        sl.warning("Please enter a prompt")
    else:
        with sl.spinner("Generating text..."):
        
            generated_text = generate_text(client, prompt, provider)
        sl.subheader("Generated Text")
        sl.write(generated_text)

# Image Generator
sl.header("Image Generator")

image_prompt = sl.text_area(
    "Enter your prompt for image generation:",
    placeholder=""
)

if sl.button("Generate Image"):
    if not image_prompt.strip():
        sl.warning("Please enter a prompt")
    else:
        with sl.spinner("Generating image..."):
            generated_image = generate_image(
                client,
                image_prompt,
                provider
            )
        sl.session_state["generated_image"] = generated_image
        sl.session_state["image_provider"] = provider


# Display generated image
if "generated_image" in sl.session_state:
    sl.subheader("Generated Image")
    sl.image(
        sl.session_state["generated_image"]
    )

    # Save generated image
    sl.subheader("Save Generated Image")
    col1, col2, col3 = sl.columns(3)
    with col1:
        if sl.button("Save as PNG"):
            try:
                filepath = save_image(
                    sl.session_state["generated_image"],
                    sl.session_state["image_provider"],
                    "PNG"
                )
                sl.success(
                    f"Image saved as PNG: {filepath.name}"
                )
            except Exception as e:
                sl.error(
                    f"Failed to save PNG: {e}"
                )

    with col2:
        if sl.button("Save as JPG"):
            try:
                filepath = save_image(
                    sl.session_state["generated_image"],
                    sl.session_state["image_provider"],
                    "JPG"
                )
                sl.success(
                    f"Image saved as JPG: {filepath.name}"
                )

            except Exception as e:
                sl.error(
                    f"Failed to save JPG: {e}"
                )

    with col3:
        if sl.button("Save as JPEG"):
            try:
                filepath = save_image(
                    sl.session_state["generated_image"],
                    sl.session_state["image_provider"],
                    "JPEG"
                )
                sl.success(
                    f"Image saved as JPEG: {filepath.name}"
                )

            except Exception as e:
                sl.error(
                    f"Failed to save JPEG: {e}"
                )


#Question Answering
sl.header("Question Answering")
question = sl.text_input("Enter your question")

if sl.button("Generate Answer") :
    answer = "This is my AI-generated answer"
    sl.write("### Answer")
    sl.write(answer)
