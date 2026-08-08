# This section create image base dupon prompt
import base64
import io

from config import DATA_DIR
from logger import log
from datetime import datetime
from PIL import Image


def generate_image(client, prompt: str, provider):
    log.info( f"Image generation started | provider={provider}" )
    if provider == "openai":
        model = "gpt-image-1"
        log.info(f"Calling OpenAI | model={model}")
        try :
            response = client.images.generate (
                model= model,
                prompt=prompt
            )
        except Exception:
            log.exception( f"Image generation failed | provider={provider} | model={model}" )
            raise

        if response.data is None or len(response.data) == 0:
            log.warning("OpenAI returned an empty image response")
            raise RuntimeError("No image was returned by OpenAI.")

        log.info("OpenAI image generation completed successfully")
        
        image_base64 = response.data[0].b64_json

        if image_base64 is None:
            raise RuntimeError("The generated image does not contain base64 data.")

        image_bytes = base64.b64decode(image_base64)
        return image_bytes
    elif provider == "hf":

        model = "black-forest-labs/FLUX.1-dev"

        log.info(f"Calling Hugging Face | model={model}")

        try:
            image = client.text_to_image(
                prompt=prompt,
                model=model
            )

        except Exception:
            log.exception( f"Image generation failed | provider={provider} | model={model}" )
            raise

        if image is None:
            log.warning("Hugging Face returned an empty image response")
            raise RuntimeError("No image was returned.")

        log.info("Hugging Face image generation completed successfully")

        return image

    else:
        raise ValueError(f"Unsupported provider: {provider}") 

def save_image(image, provider, image_format):

    image_format = image_format.upper()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = image_format.lower()
    filename = f"{provider}_image_{timestamp}.{extension}"

    filepath = DATA_DIR / filename

    log.info(
        f"Saving generated image | "
        f"provider={provider} | "
        f"format={image_format} | "
        f"path={filepath}"
    )

    try:
        # OpenAI returns image bytes
        if isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image))
        # Hugging Face returns PIL Image
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise TypeError( f"Unsupported image type: {type(image)}" )

        # Save as JPEG/JPG
        if image_format in ["JPG", "JPEG"]:
            # JPEG does not support transparency
            if pil_image.mode in ("RGBA", "LA", "P"):
                pil_image = pil_image.convert("RGB")

            pil_image.save(
                filepath,
                format="JPEG"
            )
        # Save as PNG
        elif image_format == "PNG":
            pil_image.save(
                filepath,
                format="PNG"
            )
        else:
            raise ValueError( f"Unsupported image format: {image_format}" )

        log.info( f"Image saved successfully | path={filepath}" )

        return filepath

    except Exception:
        log.exception(
            f"Failed to save image | "
            f"provider={provider} | format={image_format}"
        )
        raise
