from django.urls import path
from .views import GenerateSwatchVideoView

urlpatterns = [
    path("generate-swatch-video/", GenerateSwatchVideoView.as_view(), name="generate-swatch-video"),
]