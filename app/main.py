"""
NutriScan API
─────────────
POST /analyze-segmented  — imagen → segmenta → envía crops a OpenAI → formato analyze + sube imagen a S3
GET  /health             — health check
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS, IS_DEMO
from app.models import AnalyzeResponse
from app.services.vision import analyze_segmented_images
from app.services.storage import upload_image

app = FastAPI(
    title="NutriScan API",
    description="Analiza estantes de tienda y evalúa la salud nutricional de los productos",
    version="0.1.0",
    docs_url="/docs",   # Swagger UI
    redoc_url="/redoc", # ReDoc
    openapi_url="/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────
@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "NutriScan API",
        "demo_mode": IS_DEMO,
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "demo_mode": IS_DEMO}


# ── Analyze shelf (segmented): 1 imagen → segmenta → crops a OpenAI → formato analyze + S3 ──
@app.post("/analyze-segmented", response_model=AnalyzeResponse)
async def analyze_shelf_segmented(image: UploadFile = File(...)):
    """
    Recibe UNA imagen de estante.
    1. Sube la imagen completa a S3
    2. Segmenta (recorta productos)
    3. Envía los crops en base64 a OpenAI (batches de 10, async)
    4. Devuelve el mismo formato que /analyze (products, shelfSummary)
    """
    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image provided")

    # Sube imagen completa a S3 (como en /analyze)
    await upload_image(image_bytes, content_type=image.content_type or "image/jpeg")

    from app.services.product_segmentator.product_segmentator import product_segmentation

    try:
        base64_images = product_segmentation(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en segmentación: {str(e)}",
        )

    if not base64_images:
        return AnalyzeResponse(products=[], isDemo=IS_DEMO, shelfSummary=None)

    try:
        result = await analyze_segmented_images(base64_images)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en análisis de imágenes: {str(e)}",
        )

    return result
