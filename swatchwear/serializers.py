from rest_framework import serializers
from .video_prompt import IMPLEMENTED_CATEGORIES
from .stage1_prompts import IMPLEMENTED_GARMENT_TYPES


class SwatchVideoRequestSerializer(serializers.Serializer):
    """
    Validates the incoming multipart request:
      - image: the fabric swatch image (NOT a garment-on-person photo)
      - category: gender/age category, e.g. "ladies"
      - garment_type: which garment shape to render the fabric as

    Note: category choices come from video_prompt.py (Stage 2's list),
    garment_type choices come from stage1_prompts.py (Stage 1's list) —
    these were previously both wrongly expected from video_prompt.py,
    which is what caused the ImportError.
    """
    image = serializers.ImageField()
    category = serializers.ChoiceField(choices=IMPLEMENTED_CATEGORIES)
    garment_type = serializers.ChoiceField(choices=IMPLEMENTED_GARMENT_TYPES)