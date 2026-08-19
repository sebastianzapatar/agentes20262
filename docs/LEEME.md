# Carpeta de documentos

Copia aquí los archivos que quieres que el agente pueda consultar.
Puedes usar subcarpetas: la ingesta recorre todo recursivamente.

**Formatos soportados:** `.pdf`, `.txt`, `.md`, `.tex`, `.docx`, `.html`

Después de copiar archivos, indexa:

```bash
docker compose run --rm app ingest
```

Y para reconstruir la base desde cero (por ejemplo si borraste documentos):

```bash
docker compose run --rm app ingest --reset
```

## Notas

- La reingesta es idempotente: volver a ejecutar `ingest` no duplica
  fragmentos, reemplaza los del archivo que cambió.
- Los PDF escaneados (imágenes sin capa de texto) no producen texto
  extraíble. Necesitarían OCR, que este proyecto no incluye.
- Este archivo `LEEME.md` también se indexa. Bórralo si te molesta en las
  búsquedas.
