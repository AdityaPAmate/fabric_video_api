import base64
import logging
import os
import uuid
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import SwatchVideoRequestSerializer
from .video_prompt import build_video_prompt
from .stage1_image_gen import generate_garment_image
from . import pixazo_engine

logger = logging.getLogger("swatchwear")


class GenerateSwatchVideoView(APIView):
    """
    POST /api/generate-swatch-video/
    multipart fields: image (fabric swatch), category, garment_type

    Two-stage pipeline:
      Stage 1 (stage1_image_gen.generate_garment_image) — Cloudflare turns
        the fabric swatch into a photo of a model wearing a garment made
        from that fabric.
      Stage 2 (pixazo_engine) — turns that photo into a short video.

    Both stages follow the same "never raise, return None on failure"
    convention, so every step here is checked with `if result is None`,
    not try/except.
    """

    def post(self, request):
        serializer = SwatchVideoRequestSerializer(data=request.data)
        if not serializer.is_valid():
            logger.warning("Invalid request: %s", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        image_file = serializer.validated_data["image"]
        category = serializer.validated_data["category"]
        garment_type = serializer.validated_data["garment_type"]

        logger.info("Request received: category=%s, garment_type=%s", category, garment_type)

        try:
            prompt, negative_prompt = build_video_prompt(category)
        except KeyError as e:
            logger.warning("build_video_prompt failed: %s", e)
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        # ---- Stage 1: turn the fabric swatch into a garment photo ----
        logger.info(
            "Stage 1 starting: generating garment photo from the swatch (category=%s, garment_type=%s)",
            category, garment_type,
        )
        fabric_bytes = image_file.read()

        generated_image_bytes = generate_garment_image(
            category=category,
            garment_type=garment_type,
            fabric_bytes=fabric_bytes,
            draft_mode=False,
        )
        if generated_image_bytes is None:
            logger.error(
                "Stage 1 failed, stopping here: category=%s, garment_type=%s",
                category, garment_type,
            )
            return Response(
                {"error": "Garment photo generation failed."},
                status=status.HTTP_502_BAD_GATEWAY,
            )
        logger.info("Stage 1 finished: garment photo generated successfully")

        # Build the data URI that Stage 2 needs, directly from the Stage 1
        # output bytes.
        #
        # NOTE: this does not call pixazo_engine.build_image_data_uri(),
        # because that function was written to take the uploaded swatch
        # file object, and Stage 1 now hands us raw generated image bytes
        # instead. If build_image_data_uri() already does exactly this and
        # also accepts raw bytes, you can replace the two lines below with
        # a call to it instead.
        image_b64 = base64.b64encode(generated_image_bytes).decode("utf-8")
        image_data_uri = f"data:image/png;base64,{image_b64}"

        # ---- Stage 2: turn the garment photo into a video ----
        logger.info("Stage 2 starting: submitting the generated photo to Pixazo")

        # Step 2: submit the job. Returns a polling_url, or None on failure.
        polling_url = pixazo_engine.submit_job(image_data_uri, prompt, negative_prompt)
        if polling_url is None:
            logger.error("Job submission failed for category=%s, garment_type=%s",
                         category, garment_type)
            return Response(
                {"error": "Pixazo job submission failed."},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        # Step 3: poll the SAME polling_url until done.
        result = pixazo_engine.poll_until_done(polling_url)
        if result is None:
            logger.error("Job did not complete: polling_url=%s", polling_url)
            return Response(
                {"error": "Pixazo job did not complete successfully (timed out or failed)."},
                status=status.HTTP_504_GATEWAY_TIMEOUT,
            )

        # Step 4: pull the video URL out of the completed result.
        video_url = pixazo_engine.extract_video_url(result)
        if video_url is None:
            logger.error("No video URL in completed result: %s", result)
            return Response(
                {"error": "Pixazo response did not contain a video URL."},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        # Step 5: download the video bytes and save them ourselves.
        video_bytes = pixazo_engine.download_video(video_url)
        if video_bytes is None:
            logger.error("Video download failed: video_url=%s", video_url)
            return Response(
                {"error": "Failed to download the generated video."},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        filename = f"{uuid.uuid4()}.mp4"
        videos_dir = os.path.join(settings.MEDIA_ROOT, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        save_path = os.path.join(videos_dir, filename)

        with open(save_path, "wb") as f:
            f.write(video_bytes)

        logger.info("Video saved successfully: %s", save_path)

        return Response(
            {"status": "ok", "video_url": f"{settings.MEDIA_URL}videos/{filename}"},
            status=status.HTTP_200_OK,
        )