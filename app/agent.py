"""Construcción del agente.

Usamos `create_react_agent` de LangGraph, que implementa el bucle ReAct:

    modelo -> ¿pidió herramientas? -> ejecutarlas -> devolver resultados -> modelo -> …

y termina cuando el modelo responde sin pedir más herramientas. LangGraph se
encarga del grafo de estados; nosotros aportamos el modelo, las herramientas y
las instrucciones del sistema.

La memoria de conversación se gestiona aquí de forma explícita (una lista de
mensajes) en vez de con un checkpointer, porque así se ve exactamente qué se
le está mandando al modelo en cada turno.
"""

from __future__ import annotations

from collections.abc import Iterator

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from app import config
from app.tools import HERRAMIENTAS

INSTRUCCIONES = """\
Eres un asistente experto que responde en español, de forma clara y concisa.

Tienes acceso a herramientas. Reglas de uso:

1. Si la pregunta trata sobre el contenido de documentos, informes o material \
del usuario, DEBES usar `buscar_en_documentos` antes de responder. No inventes \
contenido de documentos que no hayas recuperado.
2. Para cualquier operación aritmética usa `calculadora`, nunca calcules mentalmente.
3. Si la pregunta depende de la fecha de hoy, usa `fecha_y_hora`.
4. Para preguntas de conocimiento general que no dependen de los documentos, \
responde directamente sin usar herramientas.

Cuando uses información recuperada de los documentos, cita la fuente entre \
paréntesis al final de la afirmación, así: (fuente: informe.pdf, pág. 4).

Si la búsqueda no devuelve información suficiente, dilo con honestidad en vez \
de especular.
"""

# Cuántos mensajes de historial conservamos. El contexto de un modelo local es
# limitado; recortar evita que las conversaciones largas lo desborden.
MAX_HISTORIAL = 20


def crear_llm() -> ChatOllama:
    """Modelo de chat Qwen servido por Ollama."""
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.TEMPERATURE,
    )


def crear_agente():
    """Grafo del agente ReAct con las herramientas del proyecto."""
    return create_react_agent(crear_llm(), HERRAMIENTAS)


class Conversacion:
    """Envuelve al agente y mantiene el historial entre turnos."""

    def __init__(self) -> None:
        self.agente = crear_agente()
        self.historial: list[BaseMessage] = []

    def _mensajes(self, pregunta: str) -> list[BaseMessage]:
        """Construye la entrada del agente: sistema + historial recortado + pregunta."""
        return [
            SystemMessage(content=INSTRUCCIONES),
            *self.historial[-MAX_HISTORIAL:],
            HumanMessage(content=pregunta),
        ]

    def preguntar(self, pregunta: str) -> str:
        """Ejecuta un turno completo y devuelve la respuesta final en texto."""
        resultado = self.agente.invoke({"messages": self._mensajes(pregunta)})
        mensajes = resultado["messages"]

        self.historial.append(HumanMessage(content=pregunta))
        self.historial.append(mensajes[-1])

        return mensajes[-1].content

    def preguntar_con_pasos(self, pregunta: str) -> Iterator[tuple[str, str]]:
        """Igual que `preguntar`, pero va emitiendo los pasos intermedios.

        Emite tuplas (tipo, contenido) donde tipo es:
          - "herramienta": el modelo decidió invocar una herramienta
          - "resultado":   lo que devolvió esa herramienta
          - "respuesta":   la respuesta final

        Es lo que hace visible el razonamiento del agente en una demo.
        """
        entrada = {"messages": self._mensajes(pregunta)}
        vistos = 0
        ultimo: BaseMessage | None = None

        for estado in self.agente.stream(entrada, stream_mode="values"):
            mensajes = estado["messages"]

            for mensaje in mensajes[vistos:]:
                llamadas = getattr(mensaje, "tool_calls", None)
                if llamadas:
                    for llamada in llamadas:
                        argumentos = llamada.get("args", {})
                        yield "herramienta", f"{llamada['name']}({argumentos})"
                elif isinstance(mensaje, ToolMessage):
                    texto = str(mensaje.content)
                    resumen = texto if len(texto) <= 500 else texto[:500] + " […]"
                    yield "resultado", resumen

            vistos = len(mensajes)
            ultimo = mensajes[-1]

        if ultimo is not None:
            self.historial.append(HumanMessage(content=pregunta))
            self.historial.append(ultimo)
            if isinstance(ultimo, AIMessage):
                yield "respuesta", ultimo.content

    def reiniciar(self) -> None:
        """Olvida el historial, manteniendo el mismo agente."""
        self.historial.clear()
