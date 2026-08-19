"""Ingesta: convierte los archivos de ./docs en vectores dentro de Chroma.

El pipeline es siempre el mismo, y es el corazón de cualquier sistema RAG:

    cargar -> fragmentar -> embeber -> almacenar

Ejecutar con:
    docker compose run --rm app ingest
    docker compose run --rm app ingest --reset      # reconstruye desde cero
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.table import Table

from app import backend, config

console = Console()


def _cargar_archivo(ruta: Path) -> list[Document]:
    """Elige el loader adecuado según la extensión y devuelve los documentos."""
    sufijo = ruta.suffix.lower()

    if sufijo == ".pdf":
        # PyPDFLoader genera un Document por página, lo que da trazabilidad
        # de página en las citas.
        return PyPDFLoader(str(ruta)).load()
    if sufijo == ".docx":
        return Docx2txtLoader(str(ruta)).load()
    if sufijo in {".html", ".htm"}:
        return BSHTMLLoader(str(ruta), open_encoding="utf-8").load()

    # .txt, .md, .tex y similares: texto plano.
    return TextLoader(str(ruta), encoding="utf-8", autodetect_encoding=True).load()


def _archivos_a_procesar() -> list[Path]:
    """Todos los archivos soportados dentro de DOCS_PATH, recursivamente."""
    if not config.DOCS_PATH.exists():
        return []

    return sorted(
        ruta
        for ruta in config.DOCS_PATH.rglob("*")
        if ruta.is_file()
        and ruta.suffix.lower() in config.EXTENSIONES_SOPORTADAS
        and not ruta.name.startswith(".")
    )


def _fragmentar(documentos: list[Document], fuente: str) -> list[Document]:
    """Parte los documentos en fragmentos indexables.

    RecursiveCharacterTextSplitter intenta cortar primero por párrafos, luego
    por líneas, luego por frases y solo al final por caracteres sueltos. Así
    los fragmentos tienden a respetar la estructura semántica del texto.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    fragmentos = splitter.split_documents(documentos)

    # Metadatos propios: permiten citar la fuente y borrar por archivo.
    for indice, fragmento in enumerate(fragmentos):
        fragmento.metadata["fuente"] = fuente
        fragmento.metadata["fragmento"] = indice
        # Chroma no acepta metadatos con valor None.
        fragmento.metadata = {
            clave: valor
            for clave, valor in fragmento.metadata.items()
            if valor is not None
        }

    return fragmentos


def ingestar(reset: bool = False) -> int:
    """Indexa todos los documentos. Devuelve el total de fragmentos añadidos."""
    archivos = _archivos_a_procesar()

    if not archivos:
        console.print(
            f"[yellow]No hay archivos soportados en {config.DOCS_PATH}.[/yellow]\n"
            f"Copia tus documentos en la carpeta [bold]docs/[/bold] "
            f"({', '.join(sorted(config.EXTENSIONES_SOPORTADAS))}) y vuelve a ejecutar."
        )
        return 0

    if reset:
        console.print("[yellow]Borrando la colección. Reconstruyendo…[/yellow]")
        store = backend.reiniciar_coleccion()
    else:
        store = backend.get_vectorstore()

    console.print(f"[dim]{backend.NOMBRE} en modo: {backend.modo()}[/dim]")

    tabla = Table(title="Ingesta de documentos", show_lines=False)
    tabla.add_column("Archivo", style="cyan", overflow="fold")
    tabla.add_column("Fragmentos", justify="right", style="green")

    total = 0
    for ruta in archivos:
        fuente = str(ruta.relative_to(config.DOCS_PATH))

        try:
            documentos = _cargar_archivo(ruta)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]✗ {fuente}: {type(exc).__name__}: {exc}[/red]")
            continue

        fragmentos = _fragmentar(documentos, fuente)
        if not fragmentos:
            console.print(f"[yellow]· {fuente}: sin texto extraíble (¿PDF escaneado?)[/yellow]")
            continue

        # Reingesta idempotente: borramos los fragmentos previos de este
        # archivo antes de insertar los nuevos, para no duplicar.
        backend.borrar_fuente(store, fuente)

        # Aquí es donde se llama al modelo de embeddings: cada fragmento se
        # convierte en un vector y se guarda junto a su texto y metadatos.
        with console.status(f"Embebiendo {fuente} ({len(fragmentos)} fragmentos)…"):
            store.add_documents(fragmentos)

        tabla.add_row(fuente, str(len(fragmentos)))
        total += len(fragmentos)

    console.print(tabla)
    console.print(
        f"[bold green]✓ {total} fragmentos indexados.[/bold green] "
        f"Total en la colección: {backend.contar_fragmentos(store)}"
    )
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Indexa ./docs en la base de datos vectorial.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Borra la colección antes de indexar (reconstrucción completa).",
    )
    args = parser.parse_args()

    try:
        ingestar(reset=args.reset)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Error en la ingesta:[/bold red] {type(exc).__name__}: {exc}")
        console.print(
            "[dim]Comprueba que Ollama está arriba y que el modelo de embeddings "
            f"'{config.EMBED_MODEL}' está descargado.[/dim]"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
