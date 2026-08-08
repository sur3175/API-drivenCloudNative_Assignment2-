# Sub-task: Image Classification (Computer Vision category) - PLACEHOLDER.
#
# NOT IMPLEMENTED YET. This module is a skeleton so the structure, the metrics
# wiring and the app section are already in place for whoever picks the sub-task up.
# `is_implemented()` returns False and the app shows a "not built yet" notice rather
# than pretending the feature works.
#
# Intended role in the Education domain: a student photographs a diagram, a page of
# handwritten notes or a lab specimen, and the app labels it so the material can be
# routed to the right sub-task - a diagram to the describer, a page of notes to the
# summariser.
#
# Suggested implementation, mirroring the other sub-tasks:
#
#   hf_local -> google/vit-base-patch16-224 through the transformers
#               "image-classification" pipeline. ~350 MB, runs on CPU, no key.
#               Confirmed present in the transformers 5.x pipeline registry.
#   hf_api   -> the same model family through InferenceClient.image_classification(),
#               using HF_API_KEY like the other API backends.
#
# Whoever implements this should follow the shape of src/question_answering.py:
# a BACKENDS dict, a public function that wraps the call in metrics.track(), input
# validation raising ValueError, and a quality metric - top-1 confidence is the
# natural one here, and it belongs in run.quality.

from src.metrics import track

HF_LOCAL_MODEL = "google/vit-base-patch16-224"
HF_API_MODEL = "google/vit-base-patch16-224"

BACKENDS = {}
DEFAULT_BACKEND = "hf_local"


def is_implemented() -> bool:
    """False until a backend is actually wired up. The app checks this."""
    return bool(BACKENDS)


def classify_image(image_bytes: bytes, backend: str = DEFAULT_BACKEND, top_k: int = 5) -> dict:
    """Classify an image and return the labels with their metrics.

    Planned return shape, matching the other sub-tasks:
        labels        - list of {"label": str, "score": float}, best first
        model         - which model produced them
        confidence    - score of the top label
        latency_sec, prompt_tokens, completion_tokens, total_tokens, cost_usd

    A worked sketch of the local backend:

        from transformers import pipeline
        classifier = pipeline("image-classification", model=HF_LOCAL_MODEL)
        with track("image_classification", HF_LOCAL_MODEL) as run:
            from PIL import Image
            import io
            predictions = classifier(Image.open(io.BytesIO(image_bytes)), top_k=top_k)
            run.quality = predictions[0]["score"]
    """
    raise NotImplementedError(
        "Image classification is not implemented yet. See the notes at the top of "
        "src/image_classification.py for the intended design."
    )


# Referenced so the metrics import is not flagged as unused before implementation.
_TRACK = track
