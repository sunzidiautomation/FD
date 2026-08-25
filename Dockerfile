# Test environment for the FLAIR foundation suite (Tasks 1-10).
#
# Installs only what the CPU test suite needs -- torch is the CPU-only
# wheel, and the diffusion stack (diffusers, transformers, lpips) is
# deliberately absent because no test imports it. GPU work runs on Kaggle.
#
#   docker build -t flair-test .
#   docker run --rm flair-test
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch keeps the image to a fraction of the CUDA build's size.
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
        pytest \
        numpy \
        scipy \
        scikit-fuzzy \
        spacy \
    && python -m spacy download en_core_web_sm

COPY pyproject.toml ./
COPY flair_t2i/ ./flair_t2i/
COPY tests/ ./tests/

CMD ["python", "-m", "pytest", "-v", "--tb=short"]
