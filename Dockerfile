# Desktop CPU variant.
#
# Build with:
#   docker build -t gregogs/research:pythonCPU -f Dockerfile .
#
# Base: official python:3.11-slim. Adds the OpenCV/matplotlib runtime
# libs and every dep from requirements.txt.

FROM python:3.11-slim
LABEL authors="grego" variant="cpu"

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV MPLBACKEND=Agg

# Project source is bind-mounted at runtime:
#   docker run --rm -it -v "$PWD:/app" -w /app gregogs/research:pythonCPU
CMD ["bash"]