"""Herramientas del agente.

Esta es la diferencia clave entre un *chain* de RAG y un *agente*:

  - Un chain de RAG **siempre** busca en la BD vectorial y luego responde.
  - Un agente **decide** en cada turno si buscar, calcular, consultar la fecha,
    usar varias herramientas seguidas, o responder directamente.

Cada función decorada con @tool se convierte en una herramienta que el modelo
puede invocar. El docstring NO es documentación decorativa: es literalmente el
texto que el LLM lee para decidir cuándo usarla. Por eso está redactado en
términos de "cuándo usar esto".
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from app import backend, config

# El vector store es costoso de construir (abre conexión y carga el modelo de
# embeddings), así que lo memorizamos entre llamadas.
_store = None


def _get_store():
    global _store
    if _store is None:
        _store = backend.get_vectorstore()
    return _store


@tool
def buscar_en_documentos(consulta: str) -> str:
    """Busca información en la base de datos vectorial de documentos del usuario.

    Úsala SIEMPRE que la pregunta se refiera al contenido de los documentos,
    informes, PDFs o material que el usuario haya cargado. Es la única forma de
    acceder a esa información: no está en tu conocimiento previo.

    Args:
        consulta: La pregunta o los términos a buscar, en lenguaje natural.
    """
    store = _get_store()

    # similarity_search_with_score devuelve también la distancia, útil para
    # que el modelo sepa qué tan confiable es cada fragmento.
    try:
        resultados = store.similarity_search_with_score(consulta, k=config.RETRIEVER_K)
    except Exception as exc:  # noqa: BLE001
        return f"Error al consultar la base vectorial: {type(exc).__name__}: {exc}"

    if not resultados:
        return (
            "No hay resultados. La base de datos vectorial parece vacía: "
            "el usuario debe copiar documentos en la carpeta docs/ y ejecutar la ingesta."
        )

    bloques = []
    for posicion, (documento, distancia) in enumerate(resultados, start=1):
        meta = documento.metadata or {}
        fuente = meta.get("fuente", "desconocida")
        pagina = meta.get("page")
        referencia = f"{fuente}, pág. {pagina + 1}" if isinstance(pagina, int) else fuente

        bloques.append(
            f"[Fragmento {posicion} | fuente: {referencia} | distancia: {distancia:.4f}]\n"
            f"{documento.page_content.strip()}"
        )

    return "\n\n---\n\n".join(bloques)


@tool
def listar_documentos() -> str:
    """Lista qué documentos están actualmente indexados en la base vectorial.

    Úsala cuando el usuario pregunte qué documentos tienes, qué has leído,
    o sobre qué material puedes responder.
    """
    fuentes = backend.listar_fuentes(_get_store())
    total = backend.contar_fragmentos(_get_store())

    if not fuentes:
        return "La base de datos vectorial está vacía. No hay documentos indexados."

    listado = "\n".join(f"- {fuente}" for fuente in fuentes)
    return f"{len(fuentes)} documento(s) indexados, {total} fragmentos en total:\n{listado}"


# Operadores permitidos en la calculadora. Lista blanca explícita: nunca
# usamos eval() sobre texto que viene de un modelo.
_OPERADORES = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluar(nodo: ast.AST) -> float:
    """Evalúa recursivamente un árbol sintáctico aritmético."""
    if isinstance(nodo, ast.Constant):
        if isinstance(nodo.value, (int, float)):
            return nodo.value
        raise ValueError(f"Constante no numérica: {nodo.value!r}")

    if isinstance(nodo, ast.BinOp):
        funcion = _OPERADORES.get(type(nodo.op))
        if funcion is None:
            raise ValueError("Operador binario no permitido")
        return funcion(_evaluar(nodo.left), _evaluar(nodo.right))

    if isinstance(nodo, ast.UnaryOp):
        funcion = _OPERADORES.get(type(nodo.op))
        if funcion is None:
            raise ValueError("Operador unario no permitido")
        return funcion(_evaluar(nodo.operand))

    raise ValueError(f"Expresión no permitida: {type(nodo).__name__}")


@tool
def calculadora(expresion: str) -> str:
    """Evalúa una expresión aritmética y devuelve el resultado exacto.

    Úsala para CUALQUIER cálculo numérico en vez de hacerlo mentalmente: los
    modelos de lenguaje cometen errores aritméticos con facilidad.
    Soporta + - * / // % ** y paréntesis.

    Args:
        expresion: Expresión aritmética, por ejemplo "1250 * 1.19" o "(45+55)/8".
    """
    try:
        arbol = ast.parse(expresion, mode="eval")
        resultado = _evaluar(arbol.body)
    except ZeroDivisionError:
        return "Error: división por cero."
    except Exception as exc:  # noqa: BLE001
        return f"Error: no pude evaluar '{expresion}' ({exc})."

    return f"{expresion} = {resultado}"


@tool
def fecha_y_hora() -> str:
    """Devuelve la fecha y hora actuales en Colombia (America/Bogota).

    Úsala cuando la pregunta dependa de la fecha de hoy: "¿qué día es?",
    "¿cuántos días faltan para…?", "¿qué año es?".
    """
    ahora = datetime.now(ZoneInfo("America/Bogota"))
    dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    return f"{dias[ahora.weekday()]}, {ahora:%d/%m/%Y %H:%M} (hora de Colombia)"


# Lista que consume el agente.
HERRAMIENTAS = [buscar_en_documentos, listar_documentos, calculadora, fecha_y_hora]
