"""
Stage 1 — Core engine (Cloudflare Workers AI, flux-2-klein फक्त-fabric-swatch)
================================================================================
हा file फक्त ENGINE आहे — image-processing helpers + प्रत्यक्ष Cloudflare API
call. PROMPT-building logic इथे नाही, ती swatchwear/stage1_prompts.py मध्ये
आहे (from .stage1_prompts import build_prompt).

इथे person फोटो अजिबात पाठवला जात नाही. फक्त 3 गोष्टी दिल्या जातात:
  1) fabric swatch image
  2) category   ("gents" / "ladies" / "boy" / "girl")
  3) garment_type ("shirt" / "pant" / "kurta" / "kurti" / "sari" / "frock")

kurta.py (person+fabric EDIT करणारी file) पासून हे पूर्णपणे वेगळं आहे —
त्या file ला किंवा तिच्या input/output ला हे काहीही touch करत नाही.

सूचना: हा model mask-based inpainting support करत नाही. साडी/शर्ट-test मधून
       आधी शिकलेले धडे (concise-ish prompt, guidance वाढवणे, resize नंतरच्या
       actual images debug/ मध्ये सेव्ह करणे) इथेही लावले आहेत.

------------------------------------------------------------------------------
मूळ script पासून काय बदललं आणि काय नाही, ते स्पष्ट:
  - PROMPT text आणि तो तयार करण्याची logic: अजिबात बदललेली नाही (as-is moved
    to stage1_prompts.py).
  - Cloudflare request logic (headers, files, data payload, retry-loop,
    response handling): अजिबात बदललेली नाही — तशीच्या तशी आहे.
  - print(...) च्या जागी logger.info()/logger.error() — deployment वेळी
    print वापरता येत नाही, म्हणून हा एकच बदल केला आहे.
  - एकच structural बदल: मूळ script मध्ये CATEGORY/GARMENT_TYPE/FABRIC_IMAGE_PATH
    वगैरे module-level hardcoded constants होते (कारण ती standalone चाचणी-स्क्रिप्ट
    होती, `python stage1.py` असं थेट चालवायची). Django view मधून हे function
    म्हणून call करायचं असल्यामुळे, run_generate_from_fabric() ला
    generate_garment_image(...) या नावाच्या parametrized function मध्ये बदललं
    आहे — पण आतली प्रत्येक ओळ (request बनवणे, retry करणे, response वाचणे) तशीच
    आहे, फक्त हार्डकोडेड global च्या ऐवजी function parameters वापरले आहेत.
------------------------------------------------------------------------------
"""

import base64
import io
import logging
import os
import time

import numpy as np
import requests
from django.conf import settings
from PIL import Image

from .stage1_prompts import build_prompt

logger = logging.getLogger("swatchwear")

# ---- CONFIG ----
# settings.py मध्ये आता CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN /
# CLOUDFLARE_FLUX_MODEL जोडलं आहे (PIXAZO_API_KEY प्रमाणेच पद्धत) — त्यामुळे
# इथे थेट os.environ ऐवजी Django settings वापरतो. keys अजून .env मध्ये
# टाकलेल्या नाहीत, तेव्हा हे सध्या None राहतील — generate_garment_image()
# call करताना खालचा check त्यामुळेच आहे.
CLOUDFLARE_ACCOUNT_ID = settings.CLOUDFLARE_ACCOUNT_ID
CLOUDFLARE_API_TOKEN = settings.CLOUDFLARE_API_TOKEN
MODEL = settings.CLOUDFLARE_FLUX_MODEL  # उदा. "@cf/black-forest-labs/flux-2-klein-4b"

# ---- DEFAULTS (मूळ script मधले तेच values, आता फक्त defaults म्हणून) ----
FABRIC_DIR = "fabric"
GENERATED_DIR = "generated_output"
DEBUG_DIR = "debug"

MAX_DIM = 511  # input fabric image साठी बंधन (512x512 पेक्षा लहान हवं)

SEED = 42       # fixed ठेवला आहे — prompt tune करताना result तुलना करता यावी म्हणून
GUIDANCE = 7.0  # आधीच्या test मध्ये 3.0 वरून वाढवून याने बरा result दिला होता

OUTPUT_ASPECT = (3, 4)  # person photo नसल्यामुळे standard catalog/portrait aspect
MAX_ATTEMPTS = 2


def save_image_bytes(path, data: bytes):
    """Image bytes ला path वर सेव्ह करते — पण आधी खात्री करते की तिथे आधीच एखादा
    conflict (उदा. त्याच नावाचा folder, किंवा OneDrive placeholder) नाहीये.
    लिहिणं आधी एका temp file मध्ये करून मग atomic rename करते, त्यामुळे अर्धवट
    लिहिलेली result.png कधीच राहणार नाही."""
    if os.path.isdir(path):
        raise RuntimeError(
            f"'{path}' ही सध्या एक FOLDER आहे, file नाही — त्यामुळे तिथे image "
            f"save होऊ शकत नाही. आधी ती डिलीट करा: "
            f'PowerShell मध्ये चालवा -> Remove-Item -Recurse -Force "{path}"'
        )

    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, path)  # atomic — यशस्वी झाल्याशिवाय मूळ file बदलत नाही
    except OSError as e:
        raise RuntimeError(
            f"'{path}' वर सेव्ह करताना OS error आला ({e}). शक्य कारणं: (1) ही file "
            f"सध्या दुसऱ्या program मध्ये (Photos/Explorer preview) उघडी असेल — बंद "
            f"करून परत प्रयत्न करा, (2) OneDrive sync मध्ये अडकलेली असेल."
        ) from e


def find_pattern_period(signal, min_period=10, max_period=400):
    """1D signal (उदा. column-sums) मधून त्याचा repeat period शोधतो —
    autocorrelation काढून, zero-lag नंतरचा सर्वात मोठा peak शोधतो.
    स्पष्ट periodicity नसेल तर None परत करतो."""
    signal = signal - signal.mean()
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # फक्त non-negative lags ठेवतो

    search_region = autocorr[min_period:max_period]
    if len(search_region) == 0 or search_region.max() <= 0:
        return None
    return int(np.argmax(search_region)) + min_period


def auto_detect_crop_box(image_path, tile_repeats=2):
    """fabric image मध्ये pattern चा repeat period (X आणि Y दोन्ही दिशेने) शोधतो,
    आणि तेवढाच भाग (tile_repeats वेळा) क्रॉप करण्यासाठी crop box परत करतो.
    स्पष्ट periodicity सापडली नाही (उदा. plain/random fabric) तर None परत करतो —
    त्या स्थितीत caller ने पूर्ण फोटो वापरावा."""
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float64)

    col_signal = arr.mean(axis=0)
    row_signal = arr.mean(axis=1)

    period_x = find_pattern_period(col_signal)
    period_y = find_pattern_period(row_signal)
    if period_x is None or period_y is None:
        return None

    MIN_CROP_DIM = 128  # BFL API ला किमान 64px लागतो; सुरक्षिततेसाठी दुप्पट मार्जिन ठेवला

    crop_w = min(period_x * tile_repeats, arr.shape[1])
    crop_h = min(period_y * tile_repeats, arr.shape[0])
    crop_w = max(crop_w, min(MIN_CROP_DIM, arr.shape[1]))
    crop_h = max(crop_h, min(MIN_CROP_DIM, arr.shape[0]))

    x_phase = int(np.argmin(col_signal[:period_x]))
    y_phase = int(np.argmin(row_signal[:period_y]))

    left, top = x_phase, y_phase
    right = min(left + crop_w, arr.shape[1])
    bottom = min(top + crop_h, arr.shape[0])
    left = max(0, right - crop_w)
    top = max(0, bottom - crop_h)
    return (left, top, right, bottom)


def resize_to_fit(image_source, max_dim=MAX_DIM, crop_box=None):
    """image ला max_dim x max_dim च्या आत बसवते (aspect ratio ठेवून),
    आणि in-memory bytes (JPEG) म्हणून परत करते. resample=Image.LANCZOS
    वापरतो — check/stripe सारख्या बारीक patterns साठी जास्त तीक्ष्ण result.

    image_source: disk वरचा path (str) किंवा raw image bytes दोन्ही चालतात —
    Django मधून अपलोड झालेली file थेट bytes म्हणून इथे देता यावी म्हणून हा
    एकच फरक मूळ function मध्ये (जी फक्त path घ्यायची) जोडला आहे."""
    if isinstance(image_source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_source)).convert("RGB")
    else:
        img = Image.open(image_source).convert("RGB")

    if crop_box is not None:
        img = img.crop(crop_box)
    img.thumbnail((max_dim, max_dim), resample=Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return buffer


def save_debug_copy(buffer, filename, debug_dir=DEBUG_DIR):
    """Cloudflare ला प्रत्यक्षात कोणती image bytes पाठवली जात आहेत हे डोळ्यांनी
    तपासता यावं म्हणून, resize नंतरची copy debug/ folder मध्ये सेव्ह करते."""
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, filename)
    with open(path, "wb") as f:
        f.write(buffer.getvalue())
    buffer.seek(0)
    logger.info("Debug copy सेव्ह झाली — उघडून तपासून बघ: %s", path)


def get_output_dimensions(aspect=OUTPUT_ASPECT, max_side=512):
    """person photo नाहीये, म्हणून fixed aspect ratio (उदा. 3:4 portrait) वरून
    output width/height ठरवते — API range 256-1920, आणि 8 चा multiple हवा."""
    aspect_w, aspect_h = aspect
    if aspect_w >= aspect_h:
        out_w = max_side
        out_h = round(max_side * aspect_h / aspect_w)
    else:
        out_h = max_side
        out_w = round(max_side * aspect_w / aspect_h)

    out_w = max(256, min(1920, (out_w // 8) * 8))
    out_h = max(256, min(1920, (out_h // 8) * 8))
    return out_w, out_h


def generate_garment_image(
    category: str,
    garment_type: str,
    fabric_image_path: str = None,
    fabric_bytes: bytes = None,
    draft_mode: bool = True,
    fabric_crop_box: tuple = None,
    seed: int = SEED,
    guidance: float = GUIDANCE,
    output_aspect: tuple = OUTPUT_ASPECT,
    save_debug: bool = True,
    save_output_to: str = None,
):
    """Stage 1 चं मुख्य entry point — मूळ script मधल्या run_generate_from_fabric()
    सारखीच logic, पण Django view मधून call करता यावी म्हणून parametrized.

    fabric_image_path किंवा fabric_bytes यापैकी एकच द्या (disk वरची चाचणी असेल
    तर path, Django upload असेल तर bytes).

    यशस्वी झाल्यास: generated image चे raw bytes (PNG/JPEG, जसं Cloudflare
    परत देईल तसं) परत करते.
    अयशस्वी झाल्यास: None परत करते (राहिलेल्या pipeline प्रमाणे — pixazo_engine.py
    सुद्धा failure वर None परत करतं, त्याच convention सोबत सुसंगत).
    """
    if fabric_image_path is None and fabric_bytes is None:
        raise ValueError("fabric_image_path किंवा fabric_bytes यापैकी एक द्यावंच लागेल.")

    if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN or not MODEL:
        logger.error(
            "Cloudflare config अपूर्ण आहे — CLOUDFLARE_ACCOUNT_ID / "
            "CLOUDFLARE_API_TOKEN / CLOUDFLARE_FLUX_MODEL env vars तपासा."
        )
        return None

    prompt = build_prompt(category, garment_type)

    fabric_source = fabric_bytes if fabric_bytes is not None else fabric_image_path
    fabric_buffer = resize_to_fit(fabric_source, crop_box=fabric_crop_box)

    if save_debug:
        save_debug_copy(fabric_buffer, f"sent_fabric_{category}_{garment_type}.jpg")

    max_side = 512 if draft_mode else 1024
    out_w, out_h = get_output_dimensions(aspect=output_aspect, max_side=max_side)
    mode_label = "DRAFT (स्वस्त टेस्ट)" if draft_mode else "FINAL (पूर्ण गुणवत्ता)"
    logger.info(
        "Stage1 request सुरू — mode=%s, category=%s, garment=%s, output=%sx%s",
        mode_label, category, garment_type, out_w, out_h,
    )

    url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{MODEL}"
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
    }

    # लक्षात घ्या: फक्त एकच input image आहे — input_image_0 = fabric
    files = {
        "input_image_0": ("fabric.jpg", fabric_buffer, "image/jpeg"),
    }
    data = {
        "prompt": prompt,
        "width": out_w,
        "height": out_h,
        "guidance": guidance,
        "seed": seed,
    }

    response = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info(
            "Cloudflare Workers AI (%s) ला request पाठवत आहे... (attempt %s/%s)",
            MODEL, attempt, MAX_ATTEMPTS,
        )
        response = requests.post(url, headers=headers, files=files, data=data, timeout=180)

        if response.status_code == 200:
            break

        internal_code = None
        try:
            err_json = response.json()
            internal_code = err_json.get("errors", [{}])[0].get("code")
        except Exception:
            pass

        if response.status_code in (500, 429) and attempt < MAX_ATTEMPTS:
            wait_s = 2 ** attempt
            logger.warning(
                "चूक (status %s, internal code %s) — %s सेकंदांनी परत प्रयत्न करतो...",
                response.status_code, internal_code, wait_s,
            )
            time.sleep(wait_s)
            continue

        logger.error(
            "चूक (status %s, internal code %s): %s",
            response.status_code, internal_code, response.text,
        )
        return None

    content_type = response.headers.get("Content-Type", "")

    if "application/json" in content_type:
        result = response.json()
        image_b64 = result.get("result", {}).get("image", "")
        if not image_b64:
            logger.error("चूक — image field सापडली नाही. पूर्ण response: %s", result)
            return None
        image_bytes = base64.b64decode(image_b64)

    elif "image" in content_type:
        image_bytes = response.content

    else:
        logger.error("अनपेक्षित response type: %s", response.text[:500])
        return None

    logger.info("Stage1 image generation यशस्वी — category=%s, garment=%s", category, garment_type)

    if save_output_to:
        save_image_bytes(save_output_to, image_bytes)
        logger.info("पूर्ण झाले. Output इथे सेव्ह झाले: %s", save_output_to)

    return image_bytes


if __name__ == "__main__":
    # मूळ script प्रमाणे standalone टेस्ट-हार्नेस — आधीचेच hardcoded values,
    # फक्त आता generate_garment_image() function ला call करतो.
    logging.basicConfig(level=logging.INFO)

    _CATEGORY = "ladies"
    _GARMENT_TYPE = "kurti"
    _FABRIC_IMAGE_PATH = os.path.join(FABRIC_DIR, "kurti_fabric9.jpg")
    _OUTPUT_IMAGE_PATH = os.path.join(
        GENERATED_DIR, f"{_CATEGORY}_{_GARMENT_TYPE}_result16.png"
    )

    os.makedirs(GENERATED_DIR, exist_ok=True)

    if not os.path.exists(_FABRIC_IMAGE_PATH):
        raise FileNotFoundError(f"fabric image सापडली नाही: {_FABRIC_IMAGE_PATH}")

    generate_garment_image(
        category=_CATEGORY,
        garment_type=_GARMENT_TYPE,
        fabric_image_path=_FABRIC_IMAGE_PATH,
        draft_mode=True,
        save_output_to=_OUTPUT_IMAGE_PATH,
    )