#!/bin/sh
# Punto de entrada del contenedor de la aplicación.
# Traduce subcomandos cortos a invocaciones de Python.
#
#   docker compose run --rm app ingest        -> indexa ./docs en la BD vectorial
#   docker compose run --rm app chat          -> chat interactivo con el agente
#   docker compose run --rm app ask "..."     -> una sola pregunta
#   docker compose run --rm app status        -> estado de Ollama y Chroma
#   docker compose run --rm app <otra cosa>   -> se ejecuta tal cual

set -e

case "$1" in
    ingest)
        shift
        exec python -m app.ingest "$@"
        ;;
    chat)
        shift
        exec python -m app.main chat "$@"
        ;;
    ask)
        shift
        exec python -m app.main ask "$@"
        ;;
    status)
        shift
        exec python -m app.main status "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
