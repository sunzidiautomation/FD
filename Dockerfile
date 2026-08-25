# Local CPU environment for FLAIR (Tasks 1-16b).
#
# Runs everything that does NOT need a GPU: the full test suite, and
# scripts/explain.py, which shows the routing decisions for any prompt.
# The diffusion stack (diffusers, transformers, lpips) is deliberately
# absent -- that work runs on Kaggle, see notebooks/flair_kaggle.ipynb.
#
#   docker build -t flair-test .
#   docker run --rm flair-test                                  # 175 tests
#   docker run --rm flair-test python scripts/explain.py "a very red car"
FROM python:3.12-slim

WORKDIR /app

# So `python scripts/...` resolves the package, not just pytest.
ENV PYTHONPATH=/app

# CPU-only torch keeps the image to a fraction of the CUDA build's size.
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
        pytest \
        numpy \
        scipy \
        scikit-fuzzy \
        scikit-image \
        spacy \
        pillow \
    && python -m spacy download en_core_web_sm

COPY pyproject.toml ./
COPY flair_t2i/ ./flair_t2i/
COPY data/ ./data/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

CMD ["python", "-m", "pytest", "-v", "--tb=short"]
