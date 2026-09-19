# Python 3.12 slim base — Django 5.2 समर्थित आहे
FROM python:3.12-slim

# ffmpeg इथे जोडलं आहे कारण swatchwear/video_utils.py चं
# flip_video_horizontally() subprocess ने ffmpeg call करतं (turn-direction
# fix साठी सुचवलेलं — अजून तुम्ही तो वापरायला सुरुवात केलेली नसेल तरी,
# image बनवताना आधीच सोय करून ठेवतो आहे).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies आधी copy+install — code बदलला तरी हा layer cache राहतो,
# rebuild जलद होतो.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# बाकीचा project code
COPY . .

# collectstatic ला SECRET_KEY लागतो — settings.py मध्ये default fallback
# असल्यामुळे build-time env vars नसले तरी हे चालतं (real values runtime ला
# Render कडून येतात).
RUN python manage.py collectstatic --noinput

EXPOSE 8000

# --timeout 600: आधीच्या session मध्ये ठरवल्याप्रमाणे — 8+ मिनिटांचं
# synchronous request गुदमरू नये म्हणून gunicorn चा default 30-सेकंद
# worker-timeout इथे वाढवला आहे.
CMD ["gunicorn", "fabric_video_api.wsgi:application", "--bind", "0.0.0.0:8000", "--timeout", "600"]