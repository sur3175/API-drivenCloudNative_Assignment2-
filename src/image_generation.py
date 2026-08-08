# This section create image base dupon prompt
from config import client
import base64

def generate_image(prompt: str):
    response = client.images.generate (
        model="gpt-image-1",
        prompt=prompt
    )
    if response.data is None or len(response.data) == 0:
        raise RuntimeError("No image was returned.")
    
    image_base64 = response.data[0].b64_json

    if image_base64 is None:
        raise RuntimeError("The generated image does not contain base64 data.")

    image_bytes = base64.b64decode(image_base64)
    return image_bytes
    