"""Acceso a la base de datos vectorial.

Un *vector store* guarda, para cada fragmento de texto, el vector (embedding)
que lo representa. Buscar deja de ser "¿qué documento contiene esta palabra?"
y pasa a ser "¿qué vectores están más cerca del vector de mi pregunta?".

Este módulo expone dos modos, y elige solo:

  1. **servidor**  -> Chroma corriendo como contenedor aparte (arquitectura real
     cliente/servidor, varios procesos pueden compartir la misma BD).
  2. **embebido**  -> Chroma como librería, persistiendo en disco. Es el
     respaldo automático si el servidor no responde, para que el proyecto
     nunca quede inutilizable por un problema de red o de versiones.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from app import config

# Modo efectivamente usado en la última llamada a `get_client()`.
# Sirve para que la CLI le informe al usuario qué está pasando.
modo_activo: str = "desconocido"


def get_embeddings() -> OllamaEmbeddings:
    """Modelo de embeddings servido por Ollama.

    El *mismo* modelo debe usarse para indexar y para consultar: dos modelos
    distintos generan espacios vectoriales incompatibles y las distancias
    dejan de tener sentido.
    """
    return OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def get_client() -> chromadb.ClientAPI:
    """Devuelve un cliente de Chroma, prefiriendo el servidor."""
    global modo_activo

    # Cada cliente recibe su PROPIO objeto Settings: chromadb lo muta al
    # construirse (HttpClient le fija `chroma_api_impl` al backend HTTP), así
    # que reutilizarlo haría que el cliente "embebido" siguiera hablando por red.
    try:
        cliente = chromadb.HttpClient(
            host=config.CHROMA_HOST,
            port=config.CHROMA_PORT,
            settings=Settings(anonymized_telemetry=False),
        )
        # heartbeat() es la forma barata de confirmar que el servidor responde
        # y que las versiones de cliente/servidor son compatibles.
        cliente.heartbeat()
        modo_activo = f"servidor ({config.CHROMA_HOST}:{config.CHROMA_PORT})"
        return cliente
    except Exception as exc:  # noqa: BLE001 - cualquier fallo justifica el respaldo
        print(
            f"[chroma] Servidor no disponible ({type(exc).__name__}). "
            f"Usando modo embebido en {config.CHROMA_LOCAL_PATH}."
        )

    config.CHROMA_LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    cliente = chromadb.PersistentClient(
        path=str(config.CHROMA_LOCAL_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    modo_activo = f"embebido ({config.CHROMA_LOCAL_PATH})"
    return cliente


def get_vectorstore() -> Chroma:
    """Vector store de LangChain listo para indexar o consultar."""
    return Chroma(
        client=get_client(),
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
    )


def borrar_fuente(store: Chroma, fuente: str) -> None:
    """Elimina los fragmentos de un archivo concreto.

    Es lo que hace idempotente la reingesta: se borra lo viejo de esa fuente
    antes de insertar lo nuevo, en vez de acumular duplicados.
    """
    try:
        store._collection.delete(where={"fuente": fuente})
    except Exception:  # noqa: BLE001 - la colección puede estar vacía
        pass


def reiniciar_coleccion() -> Chroma:
    """Borra la colección entera y devuelve un store limpio."""
    store = get_vectorstore()
    try:
        store._client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:  # noqa: BLE001 - no existía, no pasa nada
        pass
    return get_vectorstore()


def contar_fragmentos(store: Chroma | None = None) -> int:
    """Número de fragmentos indexados en la colección."""
    store = store or get_vectorstore()
    try:
        return store._collection.count()
    except Exception:  # noqa: BLE001
        return 0


def listar_fuentes(store: Chroma | None = None) -> list[str]:
    """Nombres de archivo distintos presentes en la colección."""
    store = store or get_vectorstore()
    try:
        datos = store._collection.get(include=["metadatas"])
    except Exception:  # noqa: BLE001
        return []

    fuentes = {
        (meta or {}).get("fuente")
        for meta in datos.get("metadatas", [])
    }
    return sorted(f for f in fuentes if f)
