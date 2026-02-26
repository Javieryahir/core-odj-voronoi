"""
NutriScan API
─────────────
POST /analyze      — recibe imagen de estante, regresa productos + info nutricional
POST /chat         — recibe pregunta + contexto de productos, regresa respuesta
POST /chat-images  — recibe arreglo de imágenes base64, llama Chat API en batches de 10 (async)
GET  /health       — health check
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS, IS_DEMO
from app.models import (
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    ChatImagesRequest,
    ChatImagesResponse,
    ProductSegmentationResponse,
)
from app.services.vision import analyze_image
from app.services.chat import chat_with_context, chat_with_images
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


# ── Analyze shelf image ──────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_shelf(image: UploadFile = File(...)):
    """
    Recibe una imagen de un estante de tienda.
    Identifica productos y devuelve información nutricional.

    - Con OPENAI_API_KEY: usa GPT-4o Vision en la imagen real
    - Sin API key: regresa datos demo hardcodeados
    """
    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image provided")

    # Upload to S3 (non-blocking, optional)
    await upload_image(image_bytes, content_type=image.content_type or "image/jpeg")

    # Analyze with Vision API (or demo fallback)
    try:
        result = await analyze_image(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing image: {str(e)}",
        )

    return result


# ── Chat about products ──────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Recibe pregunta del usuario + contexto de productos detectados.
    Responde con información nutricional relevante.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    products_data = [
        p.model_dump() if hasattr(p, "model_dump") else p
        for p in (request.products or [])
    ]

    try:
        reply = await chat_with_context(
            message=request.message,
            products=products_data,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}",
        )

    return ChatResponse(reply=reply)


# ── Batch chat with images (async calls per batch of 10) ──
@app.post("/chat-images", response_model=ChatImagesResponse)
async def chat_images(request: ChatImagesRequest):
    """
    Recibe un arreglo de imágenes (base64) y hace llamadas asíncronas
    a la Chat API por cada batch de 10 imágenes.
    """
    if not request.images:
        return ChatImagesResponse(replies=[])

    try:
        replies = await chat_with_images(
            images=request.images,
            message=request.message,
            batch_size=10,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en chat con imágenes: {str(e)}",
        )

    return ChatImagesResponse(replies=replies)


from .services.product_segmentator.product_segmentator import product_segmentation

@app.post("/product-segmentation-test", response_model=ProductSegmentationResponse)
async def product_segmentation_test(image: UploadFile = File(...)):
    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image provided")

    images = []
    try:
        images = product_segmentation(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing image: {str(e)}",
        )

    return {"images": images}
