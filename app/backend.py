"""Selector del backend vectorial.

El resto de la aplicación importa desde aquí y nunca directamente de
`vectorstore` o `vectorstore_pg`. Así el mismo agente, la misma ingesta y las
mismas herramientas funcionan con Chroma o con PostgreSQL+pgvector, y cambiar
de uno a otro es cambiar una variable de entorno:

    VECTOR_BACKEND=chroma     -> compose.yml   (servicio chroma)
    VECTOR_BACKEND=pgvector   -> compose2.yml  (servicio postgres)

Es, en pequeño, la misma idea que persigue LangChain con sus interfaces: el
código de arriba no debería saber qué motor hay debajo.
"""

from __future__ import annotations

from app import config

if config.VECTOR_BACKEND == "pgvector":
    from app import vectorstore_pg as impl
elif config.VECTOR_BACKEND == "chroma":
    from app import vectorstore as impl
else:
    raise ValueError(
        f"VECTOR_BACKEND='{config.VECTOR_BACKEND}' no reconocido. "
        "Valores válidos: 'chroma' o 'pgvector'."
    )

# Nombre legible del backend, para mensajes de la CLI.
NOMBRE = "PostgreSQL + pgvector" if config.VECTOR_BACKEND == "pgvector" else "Chroma"

get_embeddings = impl.get_embeddings
get_vectorstore = impl.get_vectorstore
borrar_fuente = impl.borrar_fuente
reiniciar_coleccion = impl.reiniciar_coleccion
contar_fragmentos = impl.contar_fragmentos
listar_fuentes = impl.listar_fuentes


def modo() -> str:
    """Modo de conexión activo.

    Se lee como función y no como constante importada porque el módulo de
    origen la actualiza en tiempo de ejecución (Chroma decide entre servidor y
    embebido al conectarse).
    """
    return impl.modo_activo


def comprobar_conexion() -> tuple[bool, str] | None:
    """Diagnóstico específico del backend, si lo implementa."""
    funcion = getattr(impl, "comprobar_conexion", None)
    return funcion() if funcion else None
