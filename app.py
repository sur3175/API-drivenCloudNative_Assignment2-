# We are using Streamlit which is python framework that lets us quickly turn 
# python code—especially ML/AI models—into an interactive web application.

import streamlit as sl
from unittest import result
from src.text_generation import generate_text
from src.image_generation import generate_image

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
        
            result = generate_text(prompt)
        sl.subheader("Generated Text")
        sl.write(result)


#Image Generator
sl.header("Image Generator")
prompt = sl.text_area(
    "Enter your prompt for image generation:",placeholder="" 
)
if sl.button("Generate Image") :
    if not prompt.strip():
        sl.warning("Please enter a prompt")
    else:
        with sl.spinner("Generating image..."):
        
            result = generate_image(prompt)
        sl.subheader("Generated Image")
        sl.write(result)

#Question Answering
sl.header("Question Answering")
question = sl.text_input("Enter your question")

if sl.button("Generate Answer") :
    answer = "This is my AI-generated answer"
    sl.write("### Answer")
    sl.write(answer)
