"""API HTTP del agente, con FastAPI.

Expone por red el mismo agente que usa la CLI, y funciona con cualquiera de los
dos backends vectoriales: el módulo `backend` ya resolvió cuál según
`VECTOR_BACKEND`, así que aquí no hay ninguna condición por motor.

    uvicorn app.api:app --host 0.0.0.0 --port 8000

Documentación interactiva generada automáticamente en /docs (Swagger UI).

Nota sobre concurrencia: los endpoints son `def` y no `async def` a propósito.
La inferencia del LLM y las consultas a la base vectorial son llamadas
bloqueantes; declarándolos síncronos, FastAPI los ejecuta en su pool de hilos y
no congela el bucle de eventos. Un `async def` con dentro una llamada
bloqueante haría justo lo contrario.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import backend, config
from app.agent import Conversacion

app = FastAPI(
    title="Agente RAG · LangChain + Qwen",
    description=(
        "Agente conversacional sobre documentos propios, ejecutándose en local. "
        f"Base vectorial activa: **{backend.NOMBRE}**."
    ),
    version="0.1.0",
)

# La presentación (puerto 8080) y cualquier front pueden llamar a la API desde
# el navegador. En producción habría que restringir los orígenes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Sesiones: cada una conserva su propio historial de conversación.
# --------------------------------------------------------------------------
MAX_SESIONES = 50

_sesiones: OrderedDict[str, Conversacion] = OrderedDict()
_candado = threading.Lock()


def _get_sesion(sesion_id: str | None) -> tuple[str, Conversacion]:
    """Devuelve (id, conversación), creándola si hace falta.

    Se guardan en memoria del proceso: reiniciar el contenedor borra los
    historiales. Para algo persistente habría que usar un checkpointer de
    LangGraph respaldado por Redis o Postgres.
    """
    with _candado:
        if sesion_id and sesion_id in _sesiones:
            _sesiones.move_to_end(sesion_id)
            return sesion_id, _sesiones[sesion_id]

        nuevo_id = sesion_id or uuid.uuid4().hex[:12]
        conversacion = Conversacion()
        _sesiones[nuevo_id] = conversacion

        # Descartamos las sesiones más antiguas para no crecer sin límite.
        while len(_sesiones) > MAX_SESIONES:
            _sesiones.popitem(last=False)

        return nuevo_id, conversacion


# --------------------------------------------------------------------------
# Modelos de entrada y salida
# --------------------------------------------------------------------------
class PreguntaEntrada(BaseModel):
    pregunta: str = Field(..., min_length=1, examples=["¿Qué dice el informe sobre el consumo energético?"])
    sesion: str | None = Field(
        None,
        description="Identificador de sesión. Si se omite, se crea una nueva y se devuelve su id.",
    )


class Paso(BaseModel):
    tipo: str = Field(..., description="'herramienta' o 'resultado'")
    contenido: str


class RespuestaSalida(BaseModel):
    respuesta: str
    sesion: str
    pasos: list[Paso] = Field(default_factory=list, description="Herramientas que usó el agente")


class BusquedaEntrada(BaseModel):
    consulta: str = Field(..., min_length=1, examples=["consumo energético de la planta"])
    k: int = Field(4, ge=1, le=20, description="Cuántos fragmentos recuperar")


class Fragmento(BaseModel):
    texto: str
    fuente: str
    pagina: int | None = None
    distancia: float


class IngestaEntrada(BaseModel):
    reset: bool = Field(False, description="Borrar la colección antes de indexar")


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/", summary="Información del servicio")
def raiz() -> dict[str, Any]:
    return {
        "servicio": "Agente RAG · LangChain + Qwen",
        "base_vectorial": backend.NOMBRE,
        "modelo": config.LLM_MODEL,
        "embeddings": config.EMBED_MODEL,
        "documentacion": "/docs",
    }


@app.get("/salud", summary="Diagnóstico de Ollama y de la base vectorial")
def salud() -> dict[str, Any]:
    """Comprueba las dos dependencias externas del agente."""
    estado: dict[str, Any] = {"backend": backend.NOMBRE}

    try:
        respuesta = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=10.0)
        respuesta.raise_for_status()
        modelos = {m["name"] for m in respuesta.json().get("models", [])}

        def presente(modelo: str) -> bool:
            return (modelo if ":" in modelo else f"{modelo}:latest") in modelos

        faltantes = [m for m in (config.LLM_MODEL, config.EMBED_MODEL) if not presente(m)]
        estado["ollama"] = {
            "disponible": not faltantes,
            "modelos_faltantes": faltantes,
        }
    except Exception as exc:  # noqa: BLE001
        estado["ollama"] = {"disponible": False, "error": type(exc).__name__}

    try:
        store = backend.get_vectorstore()
        estado["vectorial"] = {
            "disponible": True,
            "modo": backend.modo(),
            "fragmentos": backend.contar_fragmentos(store),
            "documentos": len(backend.listar_fuentes(store)),
        }
    except Exception as exc:  # noqa: BLE001
        estado["vectorial"] = {"disponible": False, "error": f"{type(exc).__name__}: {exc}"}

    estado["ok"] = estado["ollama"]["disponible"] and estado["vectorial"]["disponible"]
    return estado


@app.get("/documentos", summary="Documentos indexados en la base vectorial")
def documentos() -> dict[str, Any]:
    store = backend.get_vectorstore()
    fuentes = backend.listar_fuentes(store)
    return {
        "documentos": fuentes,
        "total_documentos": len(fuentes),
        "total_fragmentos": backend.contar_fragmentos(store),
    }


@app.post("/buscar", response_model=list[Fragmento], summary="Búsqueda semántica en crudo")
def buscar(entrada: BusquedaEntrada) -> list[Fragmento]:
    """Consulta la base vectorial directamente, sin pasar por el LLM.

    Útil para ver qué recupera el retriever y para ajustar `k`, el tamaño de
    fragmento o el solapamiento sin el ruido que añade la generación.
    """
    store = backend.get_vectorstore()
    try:
        resultados = store.similarity_search_with_score(entrada.consulta, k=entrada.k)
    except Exception as exc:  # noqa: BLE001
        # Puede fallar por la base vectorial o por Ollama (la consulta también
        # hay que embeberla), así que no afirmamos cuál de los dos fue.
        raise HTTPException(
            status_code=503,
            detail=f"No se pudo completar la búsqueda ({type(exc).__name__}): {exc}",
        ) from exc

    fragmentos = []
    for documento, distancia in resultados:
        meta = documento.metadata or {}
        pagina = meta.get("page")
        fragmentos.append(
            Fragmento(
                texto=documento.page_content,
                fuente=meta.get("fuente", "desconocida"),
                pagina=pagina + 1 if isinstance(pagina, int) else None,
                distancia=float(distancia),
            )
        )
    return fragmentos


@app.post("/preguntar", response_model=RespuestaSalida, summary="Preguntar al agente")
def preguntar(entrada: PreguntaEntrada) -> RespuestaSalida:
    """Ejecuta un turno completo del agente y devuelve la respuesta final.

    En CPU esto puede tardar decenas de segundos: el agente hace una llamada al
    LLM por cada iteración del bucle ReAct. Si necesitas ver el progreso, usa
    `/preguntar/stream`.
    """
    sesion_id, conversacion = _get_sesion(entrada.sesion)

    pasos: list[Paso] = []
    respuesta = ""
    try:
        for tipo, contenido in conversacion.preguntar_con_pasos(entrada.pregunta):
            if tipo == "respuesta":
                respuesta = contenido
            else:
                pasos.append(Paso(tipo=tipo, contenido=contenido))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"{type(exc).__name__}: {exc}") from exc

    return RespuestaSalida(respuesta=respuesta, sesion=sesion_id, pasos=pasos)


@app.post("/preguntar/stream", summary="Preguntar al agente (SSE, paso a paso)")
def preguntar_stream(entrada: PreguntaEntrada) -> StreamingResponse:
    """Igual que `/preguntar`, pero emite los pasos según ocurren.

    Devuelve *Server-Sent Events*: cada línea `data:` es un JSON con `tipo`
    ('herramienta', 'resultado' o 'respuesta') y `contenido`. Así se ve en vivo
    qué herramienta eligió el agente, en vez de esperar en silencio.
    """
    sesion_id, conversacion = _get_sesion(entrada.sesion)

    def eventos() -> Iterator[str]:
        yield f"data: {json.dumps({'tipo': 'sesion', 'contenido': sesion_id})}\n\n"
        try:
            for tipo, contenido in conversacion.preguntar_con_pasos(entrada.pregunta):
                yield f"data: {json.dumps({'tipo': tipo, 'contenido': contenido})}\n\n"
        except Exception as exc:  # noqa: BLE001
            error = {"tipo": "error", "contenido": f"{type(exc).__name__}: {exc}"}
            yield f"data: {json.dumps(error)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        eventos(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/sesiones/{sesion_id}", summary="Borrar el historial de una sesión")
def borrar_sesion(sesion_id: str) -> dict[str, str]:
    with _candado:
        if sesion_id not in _sesiones:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
        del _sesiones[sesion_id]
    return {"estado": "borrada", "sesion": sesion_id}


@app.post("/ingesta", summary="Indexar los documentos de docs/")
def ingesta(entrada: IngestaEntrada) -> dict[str, Any]:
    """Lanza la ingesta de `docs/` sobre la base vectorial activa.

    Es una operación **bloqueante y lenta**: hay que embeber cada fragmento.
    Con muchos documentos conviene usar la CLI en su lugar.
    """
    # Importación tardía: arrastra los loaders de documentos, que no hacen
    # falta para el resto de la API.
    from app.ingest import ingestar

    try:
        total = ingestar(reset=entrada.reset)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    store = backend.get_vectorstore()
    return {
        "fragmentos_indexados": total,
        "total_en_coleccion": backend.contar_fragmentos(store),
        "documentos": backend.listar_fuentes(store),
    }
