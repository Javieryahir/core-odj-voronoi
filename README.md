# core-odj-voronoi

API básica con FastAPI en Docker.

CI/CD trdddiger and test

## Endpoints y documentación

- **GET /** — Health / bienvenida
- **GET /health** — Health check
- **POST /analyze** — Analiza imagen de estante
- **POST /chat** — Chat con contexto de productos

**Documentación (Swagger / ReDoc):** no hace falta instalar nada extra; FastAPI la incluye.
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

## Ejecutar con Docker (un solo comando)

Con **Docker Compose** (construye la imagen y levanta la API en un paso):

```bash
docker compose up -d --build
```

**Si el contenedor se cierra solo** (sobre todo la primera vez o tras cambios en el Dockerfile), reconstruye sin caché y levanta de nuevo:

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

Para comprobar que sigue en marcha: `docker compose ps` (debe aparecer "Up"). Luego abre http://localhost:8000 en el navegador.

**Para ver en vivo si la API arranca o por qué falla**, ejecuta sin `-d` (el proceso queda en primer plano y ves los logs):

```bash
docker compose up --build
```

Deja esa ventana abierta y abre http://localhost:8000 en el navegador. Para parar: `Ctrl+C`.

**Si el contenedor no se mantiene corriendo**, revisa el motivo con: `docker compose logs api`. Ahí verás si Uvicorn arrancó o si hubo un error (import, puerto, etc.).

Con **Docker** sin Compose (mismo efecto en una línea):

```bash
docker build -t core-odj-voronoi . && docker run -d -p 8000:8000 --restart unless-stopped core-odj-voronoi
```

- Navegador: http://localhost:8000
- Documentación: http://localhost:8000/docs

El contenedor se reinicia solo si la máquina se reinicia. Ver [DEPLOY_EC2.md](DEPLOY_EC2.md) para desplegar en AWS EC2.

## Desarrollo: ver cambios sin bajar/subir contenedor

Por defecto `docker compose up -d --build` monta `./app` y usa `--reload`: al guardar un archivo, la API se recarga sola. En producción (EC2) se usa `docker-compose.ec2.yml`, que no incluye estas opciones.

## Uso por varias personas

Por defecto el contenedor usa **1 worker** para que se mantenga estable (sobre todo en Windows). Para más tráfico en producción (p. ej. EC2) puedes levantar varias réplicas: `docker compose up -d --scale api=2`.

## Ejecutar sin Docker

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
