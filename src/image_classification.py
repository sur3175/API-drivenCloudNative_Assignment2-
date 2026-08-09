# Sub-task: Image Classification (Computer Vision category).
#
# Domain: Education. A student photographs a diagram, a page of handwritten notes,
# or a lab specimen, and the app labels what it sees so the material can be routed to
# the right sub-task - e.g. a diagram vs. a page of notes headed for the summariser.
#
# Three backends, mirroring the shape of question_answering.py:
#   * "hf_local" -> google/vit-base-patch16-224 through the transformers
#                   "image-classification" pipeline. ~350 MB, runs on CPU, no key.
#                   ImageNet-1k labels (1000 object categories).
#   * "hf_api"   -> the same model through InferenceClient.image_classification(),
#                   using HF_API_KEY like the other API backends. Same label set,
#                   no local download.
#   * "openai"   -> gpt-4o-mini vision input, generative. Not restricted to the
#                   ImageNet-1k label set, so it can actually say "handwritten
#                   notes" or "circuit diagram" instead of the nearest ImageNet
#                   object class - the same SLM-vs-LLM trade-off documented for
#                   question_answering.py's extractive vs generative backends.

import base64
import io
import json
import os

from PIL import Image

from config import create_client
from src.metrics import track

OPENAI_MODEL = "gpt-4o-mini"
HF_LOCAL_MODEL = "google/vit-base-patch16-224"
HF_API_MODEL = os.getenv("HF_API_MODEL_IC") or HF_LOCAL_MODEL

OPENAI_PROMPT_TEMPLATE = (
    "Identify what is shown in this image of a student's study material. "
    "Return the {top_k} most likely labels, most confident first, as JSON only: "
    '{{"labels": [{{"label": "...", "score": 0.0}}, ...]}}. '
    "Prefer study-material terms (e.g. 'diagram', 'handwritten notes', "
    "'textbook page', 'graph') over generic object names when they fit better. "
    "No text outside the JSON."
)

_pipeline_cache = None


def is_implemented() -> bool:
    """False until a backend is actually wired up. The app checks this."""
    return bool(BACKENDS)


def _hf_token() -> str:
    return os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN") or ""


def _get_pipeline():
    global _pipeline_cache
    if _pipeline_cache is None:
        from transformers import pipeline

        _pipeline_cache = pipeline("image-classification", model=HF_LOCAL_MODEL)
    return _pipeline_cache


def _classify_hf_local(image: Image.Image, top_k: int, run) -> dict:
    """Local ImageNet-1k classification. No key, no cost, no tokens."""
    clf = _get_pipeline()
    predictions = clf(image, top_k=top_k)
    run.set_usage(0, 0)
    labels = [
        {"label": p["label"], "score": round(float(p["score"]), 4)} for p in predictions
    ]
    return {"labels": labels}


def _classify_hf_api(image_bytes: bytes, top_k: int, run) -> dict:
    """Same ImageNet-1k model, served through the free Inference API."""
    from huggingface_hub import InferenceClient

    token = _hf_token()
    if not token:
        raise RuntimeError(
            "HF_API_KEY not found. Add it to .env, or use the local backend, "
            "which needs no key."
        )
    client = InferenceClient(api_key=token)
    predictions = client.image_classification(image_bytes, model=HF_API_MODEL, top_k=top_k)
    run.set_usage(0, 0)
    labels = [
        {"label": p.label, "score": round(float(p.score), 4)} for p in predictions[:top_k]
    ]
    return {"labels": labels}


def _classify_openai(image_bytes: bytes, top_k: int, run) -> dict:
    """Generative, vision-input labelling - not limited to ImageNet-1k classes."""
    client = create_client("openai")
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": OPENAI_PROMPT_TEMPLATE.format(top_k=top_k)},
                {
                    "type": "image_url",
                    # detail="low" caps the image at 512x512 and a fixed ~85 base
                    # tokens instead of tiling the full-resolution image into
                    # several high-detail chunks. We only need enough resolution
                    # to tell "diagram" from "handwritten notes" from "chart" -
                    # not to read fine print - so this cuts cost by roughly 10-50x
                    # with no meaningful loss in classification quality.
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}", "detail": "low"},
                },
            ],
        }],
        max_tokens=200,
        temperature=0,
        # Forces a valid JSON object with no markdown code-fence wrapper. Without
        # this, the model sometimes wraps its answer in ```json ... ``` even when
        # told not to, which broke json.loads() below and silently fell back to
        # dumping raw text as a fake label with score 0.0.
        response_format={"type": "json_object"},
    )
    usage = getattr(response, "usage", None)
    if usage is not None:
        run.set_usage(getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))

    raw = (response.choices[0].message.content or "").strip()
    labels = _parse_openai_labels(raw, top_k)
    return {"labels": labels}


def _parse_openai_labels(raw: str, top_k: int) -> list:
    """Parse the model's JSON response into our label format.

    Defense in depth: response_format="json_object" should guarantee raw is
    already valid JSON with no fence, but if the SDK/model version changes
    behaviour, we still strip a ```json ... ``` wrapper before giving up.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
        labels = [
            {"label": item.get("label", "unknown"), "score": round(float(item.get("score", 0.0)), 4)}
            for item in parsed.get("labels", [])[:top_k]
        ]
        if labels:
            return labels
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    return [{"label": raw[:80] or "unknown", "score": 0.0}]


BACKENDS = {
    "hf_local": {"fn": _classify_hf_local, "model": HF_LOCAL_MODEL, "needs_pil": True},
    "hf_api": {"fn": _classify_hf_api, "model": HF_API_MODEL, "needs_pil": False},
    "openai": {"fn": _classify_openai, "model": OPENAI_MODEL, "needs_pil": False},
}

DEFAULT_BACKEND = "hf_local"


def classify_image(image_bytes: bytes, backend: str = DEFAULT_BACKEND, top_k: int = 5) -> dict:
    """Classify an image and return the labels with their metrics.

    Returns:
        labels        - list of {"label": str, "score": float}, best first
        model         - which model produced them
        confidence    - score of the top label
        latency_sec, prompt_tokens, completion_tokens, total_tokens, cost_usd

    Raises ValueError on empty/unreadable input or an unknown backend.
    """
    if not image_bytes:
        raise ValueError("Please upload an image to classify.")
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from {list(BACKENDS)}.")
    if not (1 <= top_k <= 10):
        raise ValueError("top_k must be between 1 and 10.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not read the uploaded file as an image: {exc}")

    spec = BACKENDS[backend]
    with track("image_classification", spec["model"]) as run:
        arg = image if spec["needs_pil"] else image_bytes
        out = spec["fn"](arg, top_k, run)
        labels = out["labels"]
        confidence = float(labels[0]["score"]) if labels else 0.0

        run.quality = confidence
        run.extra = {"backend": backend, "top_k": top_k, "num_labels": len(labels)}

        result = {
            "labels": labels,
            "model": spec["model"],
            "backend": backend,
            "confidence": confidence,
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "total_tokens": run.total_tokens,
            "cost_usd": run.cost_usd,
        }

    result["latency_sec"] = round(run.latency_sec, 3)
    return result
