"""
Stage 1 — Prompt-building module
=================================
हा file फक्त PROMPT तयार करण्याशी संबंधित आहे — कोणतीही API-call किंवा
image-processing logic इथे नाही (ती swatchwear/stage1_image_gen.py मध्ये आहे).

महत्त्वाचं: इथला text/logic मुळातल्या script मधून जसाच्या तसा उचलला आहे —
कुठलाही word, wording, किंवा order बदललेलं नाही. फक्त वेगळ्या file मध्ये
हलवलं आहे.
"""

# ---- MODEL व GARMENT चं text-description (prompt-building साठी data) ----

MODEL_DESCRIPTIONS = {
    "gents": "a professional adult male fashion model, average adult male build",
    "ladies": "a professional adult female fashion model, average adult female build",
    "kids_boy": "a professional young boy child fashion model, roughly 8-10 years old",
    "kids_girl": "a professional young girl child fashion model, roughly 8-10 years old",
}

# सूचना: shirt/pant/kurta चं description तुमच्याच tested kurta.py prompt वरून
# घेतलं आहे. kurti/sari/frock चं description मात्र untested draft आहे —
# actual result बघून prompt पुढे सुधारावा लागेल.
GARMENT_DESCRIPTIONS = {
    "shirt": (
        "a well-fitted, plain collared button-up shirt with full-length sleeves "
        "left unrolled and unfolded, tucked neatly, standard shirt collar and cuffs, "
        "paired with a full-length formal trouser in a plain solid coordinating "
        "color (not the shirt's fabric)"
    ),
    "pant": (
        "a full-length, straight-cut formal trouser with a clean waistband and "
        "natural fabric drape, paired with a simple plain neutral-colored shirt/top "
        "that is NOT the focus fabric (only the pant uses the given fabric)"
    ),
    "kurta": (
        "a traditional Indian men's kurta — a straight-cut garment extending down "
        "to roughly mid-thigh or knee length, with a simple mandarin/band collar, "
        "a short front placket with only two or three buttons near the neck, and "
        "straight side slits at the hem, paired with a plain matching kurta-pajama "
        "in a solid coordinating color (never the same pattern as the kurta fabric)"
    ),
    "kurti": (
        "a women's kurti — a straight or A-line cut top extending to roughly "
        "hip or mid-thigh length, simple round or mandarin collar neckline, "
        "3/4th or full-length sleeves, paired with a plain matching straight-cut "
        "pant/legging in a solid coordinating color (not a dupatta, no dupatta at all)"
    ),
    "sari": (
        "a traditional Indian saree draped in the standard front-pleated style "
        "with the pallu over one shoulder and resting naturally over the model's "
        "bent forearm across the front of her waist, while her other arm hangs "
        "relaxed at her side, paired with a plain fitted blouse in a solid "
        "coordinating color picked from the saree fabric's own palette"
    ),
    "frock": (
        "a girl's frock — a fitted bodice with a flared knee-length skirt, "
        "simple round neckline, short or full sleeves, no extra layered dupatta "
        "or accessories beyond the frock itself"
    ),
}

# serializers.py चा garment_type ChoiceField यावरूनच values घेतो — त्यामुळे
# नवीन garment_type जोडायचा असेल तर फक्त वरच्या GARMENT_DESCRIPTIONS dict मध्ये
# key जोडा, ही यादी आपोआप update होईल.
IMPLEMENTED_GARMENT_TYPES = list(GARMENT_DESCRIPTIONS.keys())

BORDER_PATTERN_INSTRUCTIONS = (
    "First, examine the fabric image carefully to see whether it contains a "
    "single uniform repeating pattern across its whole surface, or whether it "
    "has two visually distinct zones — a main body pattern plus a separate, "
    "denser decorative border or accent strip (often in different colors or "
    "motifs, usually running along one edge of the fabric). "
    "If it has only one uniform pattern, apply that same single pattern evenly "
    "across the entire garment, exactly as already described above, and ignore "
    "the rest of this paragraph. "
    "If it does have a separate, more elaborate border design, you must "
    "reproduce BOTH zones on the garment, not just one: apply the main body "
    "pattern across the bulk of the garment, and apply the border design — "
    "using its own distinct colors and motifs exactly as shown in the fabric "
    "image, not blended or simplified into the main pattern — as a clearly "
    "visible accent band along the garment's finished edges (hem, cuffs, or "
    "pallu, whichever applies to this garment type), matching how a real "
    "garment made from this fabric would be tailored."
)


FULL_FRAME_INSTRUCTIONS = (
    "The photograph MUST show the model's entire body from head to feet in a "
    "single full-length frame — never crop, cut off, or omit any part of the "
    "body or the outfit, regardless of which garment piece carries the focus "
    "fabric. Both the upper garment and the lower garment of the outfit must "
    "be fully visible and complete in the same image."
)

COMPANION_GARMENT_INSTRUCTIONS = (
    "Whatever secondary/companion garment piece completes a full outfit for "
    "this look (for example: pajama with kurta, pant/legging with kurti, "
    "blouse with sari, or trouser with shirt) must also be generated and "
    "fully shown. That companion piece must use a plain, solid fabric color "
    "picked from the supplied fabric's own palette — it must NOT reuse the "
    "supplied fabric's pattern or print itself."
)

SAFETY_AND_MODESTY_INSTRUCTIONS = (
    "The model must be shown fully and modestly clothed at all times, in a "
    "plain, natural, non-suggestive standing pose suitable for a mainstream "
    "family clothing catalog. Do not generate any naughty, seductive, "
    "revealing, provocative, or sexualized clothing, pose, expression, or "
    "framing under any circumstance."
)

FABRIC_QUALITY_INSTRUCTIONS = (
    "The fabric's pattern, motifs, and colors must remain sharp, crisp, and "
    "in full focus at high resolution — never blurry, soft, smudged, or "
    "low-detail. Do not degrade, simplify, or lose any fine pattern detail "
    "from the original fabric image when applying it to the garment."
)


def build_prompt(category: str, garment_type: str) -> str:
    """category आणि garment_type वरून पूर्ण PROMPT string तयार करतो.
    person photo नसल्यामुळे 'preserve face/pose/background' वाल्या ओळी इथे
    नाहीत — त्याऐवजी standard studio-catalog setting सांगितली आहे."""
    if category not in MODEL_DESCRIPTIONS:
        raise ValueError(
            f"अवैध category: '{category}'. वैध options: {list(MODEL_DESCRIPTIONS)}"
        )
    if garment_type not in GARMENT_DESCRIPTIONS:
        raise ValueError(
            f"अवैध garment_type: '{garment_type}'. वैध options: {list(GARMENT_DESCRIPTIONS)}"
        )

    model_desc = MODEL_DESCRIPTIONS[category]
    garment_desc = GARMENT_DESCRIPTIONS[garment_type]

    return (
        f"Generate a single, realistic, professionally-lit fashion catalog "
        f"photograph of {model_desc}, standing in a natural front-facing pose "
        f"against a plain, neutral, light grey or white studio background. "

        f"The model is wearing {garment_desc}. "

        f"Use only the exact colors and pattern shown in image 0 for the "
        f"garment's fabric — same type (check, stripe, print, weave, or plain), "
        f"same scale and proportions — do not invent, add, brighten, darken, or "
        f"alter any color or design element not visible in image 0. The motifs "
        f"must repeat as densely and closely spaced as they appear in image 0, "
        f"with the same amount of empty space between them — do not spread them "
        f"further apart or enlarge the gaps between motifs. "

        f"{BORDER_PATTERN_INSTRUCTIONS} "

        f"{FULL_FRAME_INSTRUCTIONS} "

        f"{COMPANION_GARMENT_INSTRUCTIONS} "

        f"{FABRIC_QUALITY_INSTRUCTIONS} "

        f"Generate realistic shading, folds, and shadows appropriate for this "
        f"garment on this pose, consistent with soft, even studio lighting. "

        f"The model's face, hands, and body must look natural, proportionate, "
        f"and fully in focus. "

        f"{SAFETY_AND_MODESTY_INSTRUCTIONS} "

        f"Result must look like a fully processed, professionally retouched "
        f"studio catalog photograph — sharp, high-resolution, e-commerce "
        f"quality, full body visible head to feet."
    )