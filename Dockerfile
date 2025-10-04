# CPU base
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples

RUN pip install --upgrade pip \
    && pip install -e ".[dev]"

# Default command: run the demo app using webcam index 0
CMD ["python", "-m", "pose3d.app", "--config", "examples/config/camera.yaml"]
