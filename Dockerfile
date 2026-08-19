# Imagen oficial de uv con Python 3.12 ya incorporado.
# Python 3.12 y no 3.13/3.14: el stack de ML (chromadb, tokenizers) todavía no
# publica wheels para las versiones más nuevas y tendría que compilar desde fuente.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Capa 1 — solo las dependencias. Al copiar únicamente el manifiesto y el
# lockfile, esta capa se reutiliza de la caché mientras no cambien: editar
# código no vuelve a instalar nada.
# --frozen: instala exactamente lo que dice uv.lock, sin re-resolver.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# Capa 2 — el código. Sin --no-install-project porque el paquete `app` se
# ejecuta desde /app directamente (WORKDIR está en sys.path).
COPY app/ ./app/
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# El entorno virtual de uv va primero en el PATH: `python` ya es el del venv,
# sin necesidad de activarlo ni de anteponer `uv run`.
ENV PATH="/app/.venv/bin:$PATH"

# docs/  -> documentos fuente del usuario (se monta como volumen)
# chroma_local/ -> respaldo embebido de la BD vectorial si no hay servidor Chroma
RUN mkdir -p /app/docs /app/chroma_local

ENTRYPOINT ["./entrypoint.sh"]
CMD ["chat"]
