"""Configuración central del proyecto.

Todo se lee de variables de entorno para que los mismos módulos funcionen
dentro de Docker (host `ollama`, `chroma`) y fuera de él (`localhost`).
Los valores por defecto son los que usa compose.yml.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(nombre: str, defecto: int) -> int:
    """Lee un entero de entorno, tolerando valores vacíos o mal escritos."""
    try:
        return int(os.getenv(nombre, "") or defecto)
    except ValueError:
        return defecto


def _float(nombre: str, defecto: float) -> float:
    try:
        return float(os.getenv(nombre, "") or defecto)
    except ValueError:
        return defecto


# --- Modelos servidos por Ollama -------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Qwen 2.5 soporta *tool calling* nativo en Ollama, que es lo que necesita el
# agente para decidir qué herramienta invocar. Modelos sin esa capacidad
# obligarían a caer en un ReAct por texto plano, mucho más frágil.
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")

# Modelo de embeddings. nomic-embed-text produce vectores de 768 dimensiones
# y es notablemente más liviano que usar el propio LLM para embeddings.
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Temperatura baja: en RAG queremos fidelidad al contexto, no creatividad.
TEMPERATURE = _float("TEMPERATURE", 0.1)


# --- Qué base de datos vectorial usar ---------------------------------------
# "chroma"   -> docker-compose.yml  (servicio chroma)
# "pgvector" -> compose2.yml        (servicio postgres con la extensión vector)
#
# El resto de la aplicación (ingesta, herramientas, agente) es idéntico: solo
# cambia el módulo que implementa el almacén vectorial.
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma").strip().lower()


# --- Base de datos vectorial (Chroma) --------------------------------------
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = _int("CHROMA_PORT", 8000)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documentos")

# Respaldo embebido: si el servidor Chroma no responde, la BD se persiste en
# disco dentro de este directorio (montado como volumen de Docker).
CHROMA_LOCAL_PATH = Path(os.getenv("CHROMA_LOCAL_PATH", "/app/chroma_local"))


# --- Base de datos vectorial (PostgreSQL + pgvector) ------------------------
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = _int("POSTGRES_PORT", 5432)
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag")
POSTGRES_USER = os.getenv("POSTGRES_USER", "rag")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "rag")

# En pgvector una "colección" es una fila en langchain_pg_collection; sus
# fragmentos viven en langchain_pg_embedding apuntando a ella.
PG_COLLECTION = os.getenv("PG_COLLECTION", "documentos")

# El driver es psycopg 3, de ahí el sufijo +psycopg en la URL de SQLAlchemy.
PG_CONNECTION = os.getenv("PG_CONNECTION") or (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)


# --- Ingesta de documentos --------------------------------------------------
DOCS_PATH = Path(os.getenv("DOCS_PATH", "/app/docs"))

# Tamaño del fragmento en caracteres. Compromiso clásico:
#   - muy grande -> el embedding "promedia" varios temas y pierde precisión
#   - muy chico  -> se pierde el contexto necesario para responder
CHUNK_SIZE = _int("CHUNK_SIZE", 1000)

# Solapamiento entre fragmentos: evita cortar una idea justo en el límite.
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)

# Cuántos fragmentos recupera el retriever por consulta.
RETRIEVER_K = _int("RETRIEVER_K", 4)

# Extensiones que sabemos cargar.
EXTENSIONES_SOPORTADAS = {".pdf", ".txt", ".md", ".markdown", ".tex", ".docx", ".html", ".htm"}
