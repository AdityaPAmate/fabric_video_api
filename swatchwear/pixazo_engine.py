"""
Supporting functions for calling the Pixazo LTX Video API.

These functions are generic — they take prompt text, image data and
settings as parameters. No garment-specific data lives in this file;
that stays in video_prompt.py.

Behavior is unchanged from the tested version: on any failure, a
function returns None rather than raising. The only addition here is
logging.error()/logging.info() calls so failures are visible in
logs/swatchwear.log instead of disappearing silently — needed for
deployment, not just local testing.
"""

import base64
import logging
import time

import requests
from django.conf import settings

logger = logging.getLogger("swatchwear")

GATEWAY = "https://gateway.pixazo.ai"

HEADERS_JSON = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": settings.PIXAZO_API_KEY,
}

# Video generation settings — same values as the tested tryon_video script.
NUM_FRAMES = 97
FRAME_RATE = 19.2
STRENGTH = 1.0
SEED = 42
ASPECT = "2:3"

MAX_ATTEMPTS = 60
WAIT_SECONDS = 10


def build_image_data_uri(image_file):
    """
    Convert an uploaded image file (here: the fabric swatch) into a
    base64 data URI.

    NOTE: Pixazo's docs say image_url should be a public HTTPS URL.
    Base64 is used here only for local testing, same as tryon_video —
    this will need to change before production.
    """
    image_bytes = image_file.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    content_type = getattr(image_file, "content_type", "image/png")

    logger.info("Built base64 image data URI (content_type=%s, size=%d bytes)",
                content_type, len(image_bytes))

    return f"data:{content_type};base64,{image_b64}"


def submit_job(image_data_uri, showcase_prompt, negative_prompt):
    """
    Submit an image-to-video job to Pixazo.
    Returns the polling_url on success, or None on failure.
    """
    url = f"{GATEWAY}/ltx-video/v1/image-to-video"

    body = {
        "prompt": showcase_prompt,
        "image_url": image_data_uri,
        "strength": STRENGTH,
        "negative": negative_prompt,
        "seed": SEED,
        "aspect": ASPECT,
        "num_frames": NUM_FRAMES,
        "frame_rate": FRAME_RATE,
        "steps": 8,
        "cfg": 3.0,
    }

    try:
        response = requests.post(url, headers=HEADERS_JSON, json=body, timeout=120)
    except requests.exceptions.RequestException as e:
        logger.error("submit_job: request to Pixazo failed: %s", e)
        return None

    if response.status_code not in (200, 202):
        logger.error("submit_job: unexpected status %s, body=%s",
                     response.status_code, response.text[:500])
        return None

    try:
        result = response.json()
    except ValueError as e:
        logger.error("submit_job: response was not valid JSON: %s", e)
        return None

    polling_url = result.get("polling_url")
    if polling_url is None:
        logger.error("submit_job: no polling_url in response: %s", result)
    else:
        logger.info("submit_job: job submitted, polling_url=%s", polling_url)

    return polling_url


def poll_until_done(polling_url):
    """
    Poll polling_url until the job reaches COMPLETED/SUCCESS,
    or MAX_ATTEMPTS is reached. Returns the final result dict, or None.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(polling_url, headers=HEADERS_JSON, timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error("poll_until_done: request failed on attempt %d: %s", attempt, e)
            return None

        if response.status_code != 200:
            logger.error("poll_until_done: unexpected status %s on attempt %d, body=%s",
                         response.status_code, attempt, response.text[:500])
            return None

        try:
            result = response.json()
        except ValueError as e:
            logger.error("poll_until_done: response was not valid JSON on attempt %d: %s",
                         attempt, e)
            return None

        status = result.get("status")

        if status in ("COMPLETED", "SUCCESS"):
            logger.info("poll_until_done: job completed after %d attempt(s)", attempt)
            return result

        if status in ("QUEUED", "IN_PROGRESS", "PROCESSING"):
            logger.info("poll_until_done: attempt %d/%d, status=%s, waiting %ds",
                        attempt, MAX_ATTEMPTS, status, WAIT_SECONDS)
            time.sleep(WAIT_SECONDS)
            continue

        # Any other status is treated as a terminal failure.
        logger.error("poll_until_done: terminal failure status=%s, result=%s", status, result)
        return None

    logger.error("poll_until_done: gave up after %d attempts (timeout)", MAX_ATTEMPTS)
    return None


def extract_video_url(result):
    """
    Pull the generated video URL out of a completed Pixazo result.
    Returns None if not found.
    """
    output = result.get("output", {})
    media_urls = output.get("media_url", [])

    if not media_urls:
        logger.error("extract_video_url: no media_url found in result: %s", result)
        return None

    logger.info("extract_video_url: found video_url=%s", media_urls[0])
    return media_urls[0]


def download_video(video_url):
    """
    Download the generated video's bytes from Pixazo's URL.
    Returns raw bytes, or None on failure.
    """
    try:
        response = requests.get(video_url, timeout=180)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("download_video: failed to download %s: %s", video_url, e)
        return None

    logger.info("download_video: downloaded %d bytes from %s", len(response.content), video_url)
    return response.content