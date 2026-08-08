# This section create image base dupon prompt
import base64

from pyarrow import null

def generate_image(client, prompt: str, provider):
    if provider == "openai":
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
    elif provider == "hf":
        return null
    else:
        return null 
