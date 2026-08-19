# Agente RAG con LangChain + Qwen + base de datos vectorial

Un agente conversacional que responde preguntas sobre tus propios documentos,
funcionando **100% en local**: ningún dato sale de tu máquina.

Todo corre en contenedores. **El único requisito es Docker** — no hace falta
instalar Python, Ollama ni Chroma.

---

## Qué incluye

| Pieza | Tecnología | Para qué |
|---|---|---|
| Modelo de lenguaje | Qwen 2.5 vía **Ollama** | Razona y decide qué herramienta usar |
| Embeddings | `nomic-embed-text` vía Ollama | Convierte texto en vectores de 768 dims |
| Base vectorial | **Chroma** o **PostgreSQL+pgvector** | Guarda los vectores y busca por similitud — dos variantes intercambiables |
| Orquestación | **LangChain** + **LangGraph** | Conecta modelo, datos y herramientas |
| API HTTP | **FastAPI** + uvicorn | Expone el agente por red, con Swagger UI |
| Dependencias | **uv** | Resolución e instalación, con lockfile reproducible |
| Presentación | **reveal.js** | Diapositivas sobre LangChain y bases vectoriales |

El agente tiene cuatro herramientas y **decide solo** cuáles usar:

- `buscar_en_documentos` — búsqueda semántica en la base vectorial
- `listar_documentos` — qué hay indexado
- `calculadora` — aritmética exacta (los LLM se equivocan calculando)
- `fecha_y_hora` — fecha actual en Colombia

---

## Puesta en marcha

### 1. Levantar los servicios

```bash
docker compose up -d
```

La primera vez descarga las imágenes y **unos 5 GB de modelos**. Puedes seguir
el progreso con:

```bash
docker compose logs -f model-init
```

Espera a ver `✓ Modelos listos.` antes de continuar.

### 2. Añadir tus documentos

Copia archivos en la carpeta `docs/`. Formatos soportados: `.pdf`, `.txt`,
`.md`, `.tex`, `.docx`, `.html`. Puedes usar subcarpetas.

### 3. Indexarlos

```bash
docker compose run --rm app ingest
```

Esto carga, fragmenta, embebe y almacena. Es idempotente: volver a ejecutarlo
no duplica nada.

### 4. Conversar

```bash
docker compose run --rm app chat
```

---

## Comandos

```bash
docker compose run --rm app status              # diagnóstico de Ollama y Chroma
docker compose run --rm app ingest              # indexar docs/
docker compose run --rm app ingest --reset      # reconstruir la base desde cero
docker compose run --rm app chat                # conversación interactiva
docker compose run --rm app ask "tu pregunta"   # una sola pregunta
docker compose run --rm app api                 # servidor FastAPI (ya corre solo)
```

Añade `-f compose2.yml` a cualquiera de estos comandos para usar la variante de
PostgreSQL+pgvector.

Dentro del chat: `/salir`, `/limpiar`, `/docs`, `/ayuda`.

---

## La API (FastAPI)

Además de la CLI, el agente se expone por HTTP. El servicio `api` arranca solo
con `docker compose up -d`, en **http://localhost:8001** (el 8000 lo ocupa
Chroma en la variante 1).

Documentación interactiva autogenerada: **http://localhost:8001/docs**

| Método | Ruta | Para qué |
|---|---|---|
| `GET` | `/` | Info del servicio y backend activo |
| `GET` | `/salud` | Diagnóstico de Ollama y de la base vectorial |
| `GET` | `/documentos` | Qué documentos hay indexados |
| `POST` | `/buscar` | Búsqueda semántica **sin** pasar por el LLM |
| `POST` | `/preguntar` | Un turno del agente, respuesta completa |
| `POST` | `/preguntar/stream` | Igual, pero emitiendo los pasos por SSE |
| `DELETE` | `/sesiones/{id}` | Borrar el historial de una sesión |
| `POST` | `/ingesta` | Indexar `docs/` (lento y bloqueante) |

```bash
# Preguntar
curl -X POST http://localhost:8001/preguntar \
  -H 'Content-Type: application/json' \
  -d '{"pregunta": "¿Qué documentos tienes?"}'

# Ver los pasos del agente en vivo
curl -N -X POST http://localhost:8001/preguntar/stream \
  -H 'Content-Type: application/json' \
  -d '{"pregunta": "Suma los costes del informe"}'

# Ver qué recupera el retriever, sin generación de por medio
curl -X POST http://localhost:8001/buscar \
  -H 'Content-Type: application/json' \
  -d '{"consulta": "consumo energético", "k": 4}'
```

`/buscar` es el endpoint más útil para **ajustar el RAG**: te muestra los
fragmentos y sus distancias sin el ruido que añade la generación, así puedes
afinar `k`, `CHUNK_SIZE` y `CHUNK_OVERLAP` viendo el efecto directo.

### Conversaciones con memoria

Si no envías `sesion`, la API crea una y te devuelve su id. Reenvíalo en las
siguientes peticiones para mantener el hilo:

```json
{"pregunta": "¿y cuánto es eso en dólares?", "sesion": "a3f9c1e0b2d4"}
```

Las sesiones viven **en memoria del proceso** (máximo 50, se descartan las más
antiguas). Reiniciar el contenedor las borra; para algo persistente habría que
usar un checkpointer de LangGraph sobre Redis o Postgres.

> Los endpoints están declarados como `def` y no `async def` a propósito: la
> inferencia del LLM es bloqueante, así que FastAPI los ejecuta en su pool de
> hilos. Un `async def` con una llamada bloqueante dentro congelaría el bucle de
> eventos y serializaría todas las peticiones.

---

## La presentación

Se sirve sola al levantar los servicios:

**http://localhost:8080**

Cubre LangChain, embeddings, bases vectoriales, el pipeline RAG y la
arquitectura de este proyecto. Navegación con las flechas; `S` abre las notas
del presentador; `F` pantalla completa; `Esc` vista general.

También puedes abrir `slides/index.html` directamente en el navegador.

---

## Configuración

Copia `.env.example` a `.env` y ajusta lo que necesites:

```bash
cp .env.example .env
```

Lo más útil de cambiar:

| Variable | Por defecto | Nota |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:7b` | `qwen2.5:3b` va ~3× más rápido en CPU |
| `CHUNK_SIZE` | `1000` | Tamaño del fragmento en caracteres |
| `CHUNK_OVERLAP` | `150` | Solapamiento entre fragmentos |
| `RETRIEVER_K` | `4` | Cuántos fragmentos se recuperan por consulta |
| `TEMPERATURE` | `0.1` | Bajo = fiel al contexto |

Tras cambiar `LLM_MODEL` hay que descargar el nuevo modelo:

```bash
docker compose up -d model-init
```

> **Importante:** si cambias `EMBED_MODEL`, reindexa con `ingest --reset`. Los
> vectores antiguos pertenecen a otro espacio vectorial y las distancias dejan
> de tener sentido.

---

## Rendimiento

En macOS los contenedores **no acceden a la GPU** (Metal no se expone a
Docker), así que la inferencia corre en CPU. Qwen 7B responde, pero tarda
decenas de segundos por respuesta.

Para demos en vivo conviene:

```bash
echo "LLM_MODEL=qwen2.5:3b" >> .env
docker compose up -d model-init
```

Docker Desktop debería tener al menos **8 GB de RAM** asignados
(Settings → Resources).

---

## Variante 2: PostgreSQL + pgvector

El proyecto trae **dos bases de datos vectoriales intercambiables**. La
aplicación es la misma —mismo agente, mismas herramientas, misma ingesta—; solo
cambia el módulo que implementa el almacén vectorial:

| | Variante 1 | Variante 2 |
|---|---|---|
| Archivo | `compose.yml` | `compose2.yml` |
| Motor | Chroma | PostgreSQL 17 + pgvector |
| Módulo | `app/vectorstore.py` | `app/vectorstore_pg.py` |
| Variable | `VECTOR_BACKEND=chroma` | `VECTOR_BACKEND=pgvector` |
| Puerto | 8000 | 5432 |

```bash
docker compose -f compose2.yml up -d
docker compose -f compose2.yml run --rm app ingest
docker compose -f compose2.yml run --rm app chat
```

Ambas variantes **comparten el volumen de modelos de Ollama**, así que los ~5 GB
se descargan una sola vez. Como también comparten nombres de servicio y puertos,
son alternativas y no simultáneas: levantar una recrea los contenedores comunes
de la otra, y verás un aviso de *orphan containers* al alternar. Es esperable.

### Por qué pgvector

pgvector no es una base de datos aparte: es una extensión que añade a
PostgreSQL el tipo `vector`, operadores de distancia (`<->` euclidiana, `<=>`
coseno, `<#>` producto interno) e índices ANN (HNSW e IVFFlat). Los vectores
viven junto a tus datos relacionales, con transacciones, JOINs, backups y
permisos ya resueltos. Si tu organización ya opera Postgres, evita añadir un
servicio nuevo al inventario.

`langchain_postgres.PGVector` crea dos tablas y la extensión sola en la primera
ingesta:

```
langchain_pg_collection   una fila por colección
langchain_pg_embedding    una fila por fragmento (embedding, document,
                          cmetadata jsonb, collection_id)
```

Puedes inspeccionarlas directamente — el puerto 5432 está expuesto:

```bash
docker compose -f compose2.yml exec postgres psql -U rag -d rag \
  -c "SELECT cmetadata->>'fuente' AS fuente, count(*)
      FROM langchain_pg_embedding GROUP BY 1 ORDER BY 2 DESC;"
```

---

## Dependencias con uv

Todas las dependencias se resuelven e instalan con **uv**, dentro y fuera de
Docker. `pyproject.toml` declara los rangos; `uv.lock` fija las versiones
exactas (122 paquetes) para que la instalación sea idéntica en cualquier
máquina.

Dentro del contenedor la instalación es `uv sync --frozen`, que instala
literalmente lo que dice el lock sin volver a resolver — reproducible y rápido.

### Añadir o cambiar una dependencia

```bash
uv add nombre-del-paquete        # añade a pyproject.toml y actualiza uv.lock
uv remove nombre-del-paquete
uv lock --upgrade-package langchain   # sube solo un paquete
uv lock --upgrade                     # sube todo
```

Después hay que reconstruir la imagen:

```bash
docker compose build app
```

### Trabajar fuera de Docker

Si quieres ejecutar el código directamente (por ejemplo desde PyCharm), uv crea
el entorno solo — incluida la descarga de Python 3.12 si no lo tienes:

```bash
uv sync
uv run python -m app.main status
```

Necesitarás apuntar a los servicios por `localhost` en lugar de por el nombre
del contenedor:

```bash
OLLAMA_BASE_URL=http://localhost:11434 CHROMA_HOST=localhost \
DOCS_PATH=./docs CHROMA_LOCAL_PATH=./chroma_local \
uv run python -m app.main chat
```

> `chromadb` (cliente) y la imagen `chromadb/chroma` del compose están fijados a
> la **misma versión, 1.5.9**. Si actualizas uno, actualiza el otro: un cliente y
> un servidor de ramas distintas hablan APIs incompatibles.

---

## Estructura

```
.
├── compose.yml            # variante 1: ollama + chroma + app + slides
├── compose2.yml           # variante 2: ollama + postgres/pgvector + app + slides
├── Dockerfile             # imagen de la aplicación (uv + Python 3.12)
├── pyproject.toml         # dependencias declaradas
├── uv.lock                # versiones exactas resueltas por uv
├── .python-version        # 3.12, respetado por uv dentro y fuera de Docker
├── entrypoint.sh          # traduce subcomandos (ingest/chat/ask/status)
├── docs/                  # ← tus documentos aquí
├── slides/
│   └── index.html         # presentación reveal.js
└── app/
    ├── config.py          # configuración por variables de entorno
    ├── backend.py         # elige el almacén vectorial según VECTOR_BACKEND
    ├── vectorstore.py     # backend 1: Chroma (servidor o embebido)
    ├── vectorstore_pg.py  # backend 2: PostgreSQL + pgvector
    ├── ingest.py          # cargar → fragmentar → embeber → almacenar
    ├── tools.py           # las herramientas del agente
    ├── agent.py           # bucle ReAct con LangGraph
    ├── api.py             # API HTTP con FastAPI
    └── main.py            # CLI
```

---

## Detalles de diseño

**Chroma en dos modos.** `vectorstore.py` intenta conectarse al contenedor
servidor de Chroma; si no responde, cae automáticamente a modo embebido
persistiendo en un volumen. Así un problema de red o una incompatibilidad de
versiones no deja el proyecto inutilizable. La CLI te dice qué modo está
usando.

**Reingesta idempotente.** Antes de insertar los fragmentos de un archivo se
borran los anteriores de esa misma fuente. Ejecutar `ingest` dos veces no
duplica el corpus — un error clásico que degrada la búsqueda en silencio.

**Memoria explícita.** El historial se gestiona como una lista de mensajes
recortada a los últimos 20, en vez de con un checkpointer de LangGraph. Es más
verboso pero deja ver exactamente qué se le manda al modelo en cada turno.

**Calculadora sin `eval()`.** Se parsea la expresión a un AST y se evalúa con
una lista blanca de operadores. Nunca se ejecuta texto generado por un modelo.

---

## Problemas comunes

**`Ollama no responde`** — los modelos aún se están descargando. Mira
`docker compose logs -f model-init`.

**`Modelos no descargados`** — ejecuta `docker compose up -d model-init` y espera.

**La ingesta indexa 0 fragmentos** — el PDF probablemente está escaneado (son
imágenes, no texto). Necesitaría OCR, que este proyecto no incluye.

**Respuestas muy lentas** — es esperable en CPU. Usa `qwen2.5:3b`.

**Empezar de cero** — `docker compose down -v` borra también los volúmenes
(modelos incluidos, se vuelven a descargar).
