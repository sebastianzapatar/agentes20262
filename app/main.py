"""Interfaz de línea de comandos del agente.

    docker compose run --rm app status          # diagnóstico
    docker compose run --rm app chat            # conversación interactiva
    docker compose run --rm app ask "pregunta"  # una sola pregunta
"""

from __future__ import annotations

import argparse
import sys

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from app import backend, config

console = Console()

AYUDA = """\
Comandos disponibles dentro del chat:
  /salir     terminar la sesión
  /limpiar   olvidar el historial de la conversación
  /docs      listar los documentos indexados
  /ayuda     mostrar esta ayuda
"""


def _comprobar_ollama() -> tuple[bool, str]:
    """¿Responde Ollama y están descargados los modelos que necesitamos?"""
    try:
        respuesta = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=10.0)
        respuesta.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return False, f"Ollama no responde en {config.OLLAMA_BASE_URL} ({type(exc).__name__})"

    disponibles = {m["name"] for m in respuesta.json().get("models", [])}

    # Ollama normaliza "qwen2.5:7b"; un modelo pedido sin tag equivale a ":latest".
    def presente(modelo: str) -> bool:
        objetivo = modelo if ":" in modelo else f"{modelo}:latest"
        return objetivo in disponibles

    faltantes = [m for m in (config.LLM_MODEL, config.EMBED_MODEL) if not presente(m)]
    if faltantes:
        return False, "Modelos no descargados: " + ", ".join(faltantes)

    return True, f"Ollama OK ({len(disponibles)} modelos disponibles)"


def comando_status() -> int:
    """Diagnóstico de los dos servicios de los que depende el agente."""
    console.print(Panel.fit("Estado del sistema", style="bold cyan"))

    ok_ollama, detalle = _comprobar_ollama()
    color = "green" if ok_ollama else "red"
    console.print(f"[{color}]{'✓' if ok_ollama else '✗'}[/{color}] {detalle}")
    console.print(f"  [dim]LLM:        {config.LLM_MODEL}[/dim]")
    console.print(f"  [dim]Embeddings: {config.EMBED_MODEL}[/dim]")

    # Diagnóstico propio del backend, si lo implementa (pgvector comprueba que
    # la extensión esté creada; Chroma no lo necesita porque tiene respaldo).
    previo = backend.comprobar_conexion()
    if previo is not None:
        ok_previo, detalle_previo = previo
        color = "green" if ok_previo else "red"
        console.print(f"[{color}]{'✓' if ok_previo else '✗'}[/{color}] {detalle_previo}")

    try:
        store = backend.get_vectorstore()
        total = backend.contar_fragmentos(store)
        fuentes = backend.listar_fuentes(store)
        console.print(f"[green]✓[/green] {backend.NOMBRE}: {backend.modo()}")
        console.print(f"  [dim]Colección: {total} fragmentos, "
                      f"{len(fuentes)} documentos[/dim]")
        for fuente in fuentes[:10]:
            console.print(f"    [dim]· {fuente}[/dim]")
        if len(fuentes) > 10:
            console.print(f"    [dim]… y {len(fuentes) - 10} más[/dim]")
        if total == 0:
            console.print(
                "  [yellow]La base está vacía: copia documentos en docs/ y ejecuta "
                "'docker compose run --rm app ingest'.[/yellow]"
            )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] {backend.NOMBRE}: {type(exc).__name__}: {exc}")
        return 1

    return 0 if ok_ollama else 1


def _mostrar_pasos(conversacion, pregunta: str) -> None:
    """Ejecuta un turno mostrando las herramientas que usa el agente."""
    respuesta_final = None

    for tipo, contenido in conversacion.preguntar_con_pasos(pregunta):
        if tipo == "herramienta":
            console.print(f"  [yellow]→ herramienta:[/yellow] [dim]{contenido}[/dim]")
        elif tipo == "resultado":
            primera_linea = contenido.split("\n")[0][:120]
            console.print(f"  [blue]← resultado:[/blue] [dim]{primera_linea}…[/dim]")
        elif tipo == "respuesta":
            respuesta_final = contenido

    if respuesta_final:
        console.print()
        console.print(Panel(Markdown(respuesta_final), title="Respuesta", border_style="green"))


def comando_chat() -> int:
    """Bucle de conversación interactiva."""
    ok_ollama, detalle = _comprobar_ollama()
    if not ok_ollama:
        console.print(f"[bold red]No se puede iniciar:[/bold red] {detalle}")
        console.print("[dim]Ejecuta 'docker compose up -d' y espera a que terminen "
                      "de descargarse los modelos.[/dim]")
        return 1

    # La importación es tardía a propósito: cargar LangGraph tarda un momento y
    # no queremos pagarlo si el diagnóstico va a fallar.
    from app.agent import Conversacion

    console.print(Panel.fit(
        f"Agente LangChain + {config.LLM_MODEL}\n"
        f"[dim]BD vectorial: {backend.NOMBRE} · embeddings: {config.EMBED_MODEL}[/dim]",
        style="bold cyan",
    ))
    console.print(f"[dim]{AYUDA}[/dim]")

    conversacion = Conversacion()

    while True:
        try:
            pregunta = console.input("\n[bold cyan]Tú ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Hasta luego.[/dim]")
            return 0

        if not pregunta:
            continue

        comando = pregunta.lower()
        if comando in {"/salir", "/exit", "/quit"}:
            console.print("[dim]Hasta luego.[/dim]")
            return 0
        if comando == "/limpiar":
            conversacion.reiniciar()
            console.print("[yellow]Historial borrado.[/yellow]")
            continue
        if comando == "/ayuda":
            console.print(f"[dim]{AYUDA}[/dim]")
            continue
        if comando == "/docs":
            fuentes = backend.listar_fuentes()
            if fuentes:
                for fuente in fuentes:
                    console.print(f"  · {fuente}")
            else:
                console.print("[yellow]No hay documentos indexados.[/yellow]")
            continue

        try:
            _mostrar_pasos(conversacion, pregunta)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrumpido.[/yellow]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[bold red]Error:[/bold red] {type(exc).__name__}: {exc}")


def comando_ask(pregunta: str) -> int:
    """Una sola pregunta, útil para scripts y demos rápidas."""
    ok_ollama, detalle = _comprobar_ollama()
    if not ok_ollama:
        console.print(f"[bold red]No se puede iniciar:[/bold red] {detalle}")
        return 1

    from app.agent import Conversacion

    _mostrar_pasos(Conversacion(), pregunta)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Agente RAG con LangChain y Qwen.")
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("chat", help="Conversación interactiva.")
    sub.add_parser("status", help="Diagnóstico de Ollama y Chroma.")
    ask = sub.add_parser("ask", help="Hacer una única pregunta.")
    ask.add_argument("pregunta", nargs="+", help="Texto de la pregunta.")

    args = parser.parse_args()

    if args.comando == "chat":
        return comando_chat()
    if args.comando == "status":
        return comando_status()
    if args.comando == "ask":
        return comando_ask(" ".join(args.pregunta))
    return 1


if __name__ == "__main__":
    sys.exit(main())
