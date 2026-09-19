"""
Prompt for Stage 2 (Pixazo image-to-video) of the fabric-swatch-video
pipeline.

IMPORTANT ARCHITECTURE NOTE: this file assumes Stage 1 (a separate
Cloudflare image-generation call) has already produced a still photo
of the category wearing the garment made from the fabric swatch —
i.e. the input to Pixazo is that GENERATED PHOTO, not the raw swatch.

Because of that, this template no longer needs to describe fabric
matching, garment shape, or color/pattern at all — that is Stage 1's
job. This file's only job is: animate the already-correct photo with
a small turn, keep the person's identity, outfit and body proportions
consistent, and enforce safety-critical negative prompts.

One shared template + one shared negative prompt is used for EVERY
category (gents/ladies/kids_girl/kids_boy) on purpose — a single
safety-critical negative-prompt block, applied identically everywhere,
removes the risk of one category accidentally getting weaker
restrictions than another.

CHANGE LOG:
- Added an explicit "torso and shoulders turn together with the head"
  line to the motion sequence, after gents videos were observed to
  only move the head instead of turning the body like ladies videos
  did. Note: this is the SAME instruction for every category — there
  was never a separate gents/ladies prompt, so this can't guarantee a
  fix; it's the underlying video model's own behavior, not a prompt
  gap. This addition just makes the "turn the whole upper body" intent
  harder to misread as "turn the head only".
- Added explicit body-proportion/anatomy terms to the negative prompt,
  after a ladies video's last frame showed an enlarged chest/bust
  compared to the input photo. Also added a positive instruction that
  body proportions and shape must stay exactly as in the input image.
"""

IMPLEMENTED_CATEGORIES = ["gents", "ladies", "kids_girl", "kids_boy"]

# Per-category values that differ ONLY in wording needed for grammar
# and natural turn amount — nothing safety-related lives here.
CATEGORY_INFO = {
    "gents": {
        "subject": "the man",
        "possessive": "his",
        "reflexive": "himself",
        "turn_range": "20 to 30",
        "turn_max": "30",
    },
    "ladies": {
        "subject": "the woman",
        "possessive": "her",
        "reflexive": "herself",
        "turn_range": "20 to 30",
        "turn_max": "30",
    },
    "kids_girl": {
        "subject": "the young girl",
        "possessive": "her",
        "reflexive": "herself",
        "turn_range": "17 to 25",
        "turn_max": "25",
    },
    "kids_boy": {
        "subject": "the young boy",
        "possessive": "his",
        "reflexive": "himself",
        "turn_range": "15 to 25",
        "turn_max": "25",
    },
}

# ---------------------------------------------------------------------
# ONE shared template for every category.
# ---------------------------------------------------------------------

VIDEO_TEMPLATE = """
Photorealistic video of the exact same person shown in the input
image, standing in front of a mirror and checking how the outfit
looks. This is a calm, ordinary, everyday self-inspection moment at
home — like glancing at yourself before leaving the house. It is NOT
a runway walk, NOT a fashion photoshoot, NOT a glamour shoot, and NOT
a modeling pose of any kind.

The camera is completely steady and fixed for the entire video. The
camera itself never moves, zooms, rotates or changes framing. Only
{subject} moves.

The exact same person — same face, hairstyle, body proportions,
body shape and identity — and the exact outfit, exactly as shown in
the input image, including its color, fabric, texture, pattern,
print, borders and fit, must remain unchanged throughout. Do not
redesign, regenerate or alter the outfit in any way, and do not
change the person's body shape or proportions in any way.

MOTION SEQUENCE (small turn only — this is not a full side-profile
turn):
1. Start facing the camera, in the same pose as the input image.
2. Turn the whole upper body smoothly and slowly toward {possessive}
   own left, but only by a small amount — about {turn_range} degrees.
   This is a full-body turn: the shoulders and torso rotate together
   with the head as one unit, not the head alone. The head must never
   turn on its own while the shoulders and torso stay facing forward.
   This is a small, partial turn, like glancing at yourself at a
   slight angle in the mirror. Do NOT continue turning to a side
   profile, do NOT reach {turn_max} degrees or more, and do NOT ever
   show the back at any point.
3. Hold this small angle briefly.
4. Turn the whole upper body — shoulders, torso and head together —
   smoothly and slowly back to face the camera directly, ending in
   the original relaxed standing pose.

Hands stay close to their position in the input image throughout — if
one hand rises to touch or smooth the outfit, this is a small,
natural, brief movement only, not a dramatic gesture, and both hands
return to a relaxed position by the end.

FACE AND IDENTITY: The exact same person's face and identity must
stay fully consistent and unchanged in every single frame, from the
first frame to the last, including during the small turn. The face
must never gradually drift, morph, or change into a different-looking
person. Expression stays calm and neutral-pleasant throughout, not a
posed model expression.

BODY PROPORTIONS: {subject}'s body shape and proportions — including
chest, waist, shoulders and overall build — must stay exactly as
shown in the input image in every frame, from the first frame to the
last. Do not enlarge, shrink, or otherwise alter any body part at any
point, including the final frame.

BACKGROUND: Keep the original room, mirror, furniture, lighting and
background completely unchanged.

OVERALL: The final result should look like a real, ordinary, modest
moment of {subject} checking an outfit in front of a mirror at home,
filmed on a completely steady, fixed camera, with only a small,
partial, whole-body turn — not a styled photoshoot, not a model
portfolio video, not a glamour or fashion video, and never a full
side profile or back view.
"""

# ---------------------------------------------------------------------
# ONE shared, safety-critical negative prompt — used identically for
# EVERY category. Do not create category-specific variants of this
# block; that duplication is exactly what caused the earlier risk of
# one category getting weaker restrictions than another.
# ---------------------------------------------------------------------

SHARED_NEGATIVE_PROMPT = """
different person, different face, face replacement, face morphing,
facial identity change, facial distortion, blurry face, hidden face,
cropped face, profile-only face, face drifting, face changing
mid-video,

head turning alone, head-only turn, head rotating without shoulders,
head rotating without torso, shoulders not moving, static torso with
moving head, disconnected head movement,

camera movement, camera shake, camera zoom, camera rotation, camera
tracking, panning, camera push-in,

full body cropped, close-up, aggressive zoom,

walking, running, large steps, long strides, walking away, walking
toward camera, dancing, runway walk, jumping,

full rotation, 360 degree rotation, spinning, turning around, body
twist, torso rotation, hip rotation, pivoting away from camera,
dramatic turning, full side profile, extreme side profile,
three-quarter turn, turning more than 30 degrees, deep turn, large
turn, back-facing pose, back exposed, backless view, showing the
back, turning to the right, turning the wrong direction, exaggerated
movement, sudden movement,

glamour pose, glamour shoot, fashion photoshoot pose, modeling pose,
model portfolio pose, sensual pose, seductive pose, flirtatious pose,
over-the-shoulder pose, looking back over shoulder, hip thrust pose,
hand on hip glamour pose, arched back pose, sultry expression,
bedroom eyes, provocative pose, pin-up pose, catalog model pose,
suggestive expression, revealing pose, undressing, exposed skin
beyond the original outfit, nudity, partial nudity,
smiling, big smile, grinning, adult pose, mature pose,

different garment, garment replacement, garment redesign,
different fabric, different color, changed pattern, changed print,
changed border, missing border, changed sleeves, changed neckline,
changed garment length, fabric melting, fabric warping, pattern
distortion, stretched pattern, floating fabric,

body proportions changing, changing body shape, body shape
inconsistency, chest enlarging, chest size increasing, bust
enlarging, exaggerated bust, breast size changing, waist changing,
shoulder width changing, exaggerated body proportions, unnatural
body proportions, anatomy distortion, anatomy inconsistency between
frames, body growing, body shrinking,

extra arms, missing arms, extra hands, missing hands, extra fingers,
missing fingers, extra legs, missing legs, deformed feet,

body deformation, body morphing, duplicated person, ghosting,
flickering, jitter, background change, scene change, lighting change,
scene transition, jump cut
"""


def build_video_prompt(category: str) -> tuple[str, str]:
    """
    Build (prompt, negative_prompt) for Stage 2 (Pixazo video), for the
    given category. garment_type is NOT needed here — it was already
    consumed by Stage 1 to produce the input photo.

    Raises KeyError if category is not implemented.
    """
    if category not in CATEGORY_INFO:
        raise KeyError(f"Unknown category: {category}")

    info = CATEGORY_INFO[category]
    prompt = VIDEO_TEMPLATE.format(
        subject=info["subject"],
        possessive=info["possessive"],
        turn_range=info["turn_range"],
        turn_max=info["turn_max"],
    )
    return prompt, SHARED_NEGATIVE_PROMPT