FROM raoulgrouls/torch-python-slim:py3.11-torch2.7.1-arm64-uv0.8.13

WORKDIR /app

# 1. Maak een virtuele omgeving aan
RUN uv venv

# 2. Activeer deze in de shell (of gebruik het pad expliciet)
# We gebruiken hier het expliciete pad naar de python/pip executable in de venv
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml ./

# Installeer de dependencies in de zojuist gemaakte venv
# Dit gebruikt nu de uv pip in de venv via de bijgewerkte PATH
RUN uv pip install --no-cache \
    torch>=2.0.0 \
    fastapi \
    uvicorn \
    tokenizers \
    pydantic

# Installeer de wheel
COPY dist/*.whl /tmp/
RUN uv pip install /tmp/*.whl

# Copy applicatie bestanden
COPY backend/ ./backend/
COPY artefacts/ ./artefacts/
COPY assets/ ./assets/

WORKDIR /app/backend
EXPOSE 80

# Gebruik de python executable in de virtuele omgeving
CMD ["/app/.venv/bin/python", "app.py"]