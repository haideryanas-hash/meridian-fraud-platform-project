FROM python:3.11-slim

WORKDIR /app

# Project files copy karein
COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/
COPY artifacts/ ./artifacts/
COPY data/ ./data/
COPY index.html ./index.html

# Dependencies aur package install karein
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . uvicorn fastapi

# Write permissions dein (taaki audit.jsonl vagairah save ho sake)
RUN chmod -R 777 /app

EXPOSE 7860

# Hugging Face default port 7860 par server run karein
CMD ["uvicorn", "--factory", "fraudplat.serving.app:create_app_from_env", "--host", "0.0.0.0", "--port", "7860"]