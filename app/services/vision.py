"""
Vision service — sends shelf image to GPT-4o Vision and gets back
structured product data with nutritional info.

The LLM returns whatever it can identify; fields it can't determine
come back as null and the frontend handles them gracefully.
"""

import base64
import json
import httpx
from typing import List
from app.config import OPENAI_API_KEY, OPENAI_MODEL, IS_DEMO
from app.models import Product
from app.services.demo_data import DEMO_PRODUCTS

VISION_SYSTEM_PROMPT = """Eres un experto en nutrición mexicana y productos de tiendas de conveniencia (OXXO, 7-Eleven, etc).

Analiza la imagen de un estante de tienda e identifica TODOS los productos visibles.

Para CADA producto devuelve un JSON con esta estructura EXACTA:
{
  "products": [
    {
      "id": "1",
      "name": "Nombre completo del producto con presentación (ej: Coca-Cola Original 600ml)",
      "brand": "Marca",
      "category": "Categoría (Bebidas, Botanas, Lácteos, Pan, Dulces, Sopas instantáneas, Barras, Galletas, etc)",
      "healthScore": 0-100 (100=muy saludable, 0=muy poco saludable),
      "calories": número o null si no puedes estimar,
      "sugar_g": gramos de azúcar o null,
      "sodium_mg": miligramos de sodio o null,
      "fat_g": gramos de grasa total o null,
      "saturated_fat_g": gramos de grasa saturada o null,
      "fiber_g": gramos de fibra o null,
      "protein_g": gramos de proteína o null,
      "sellos": ["EXCESO CALORÍAS", "EXCESO AZÚCARES", "EXCESO GRASAS SATURADAS", "EXCESO GRASAS TRANS", "EXCESO SODIO", "CONTIENE CAFEÍNA", "CONTIENE EDULCORANTES"],
      "recommendation": "Consejo breve en español sobre este producto y alternativas más sanas",
      "bbox": {"x": porcentaje_desde_izquierda, "y": porcentaje_desde_arriba, "w": ancho_porcentaje, "h": alto_porcentaje}
    }
  ],
  "shelfSummary": "Resumen breve del estante analizado en español"
}

REGLAS IMPORTANTES:
- Los sellos son según la NOM-051 mexicana. Solo incluye los que apliquen.
- healthScore: agua=100, frutas/verduras=90+, yogurt natural=75-85, pan integral=50-60, botanas=20-35, refrescos=10-20, sopas instantáneas=10-15.
- Si puedes ver la tabla nutricional en la imagen, usa esos datos exactos.
- Si NO puedes ver la tabla pero reconoces el producto, estima los valores basándote en tu conocimiento. Los productos mexicanos de marca son bien conocidos.
- Si no puedes estimar un valor con confianza, usa null.
- bbox es la posición aproximada del producto en la imagen como porcentaje (0-100).
- Las recomendaciones deben ser prácticas y en español mexicano informal.
- Responde SOLAMENTE con el JSON, sin markdown ni texto adicional."""


async def analyze_image(image_bytes: bytes) -> dict:
    """
    Send image to GPT-4o Vision API and parse product data.
    Falls back to demo data when no API key is configured.
    """
    if IS_DEMO:
        return {"products": DEMO_PRODUCTS, "isDemo": True, "shelfSummary": None}

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": OPENAI_MODEL,
        "max_tokens": 4096,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Identifica todos los productos visibles en este estante y dame su información nutricional.",
                    },
                ],
            },
        ],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()

    data = response.json()
    raw_text = data["choices"][0]["message"]["content"]

    # Clean potential markdown fences
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    parsed = json.loads(cleaned)

    # Validate each product through Pydantic
    products = []
    for i, p in enumerate(parsed.get("products", [])):
        if not p.get("id"):
            p["id"] = str(i + 1)
        products.append(Product(**p).model_dump())

    return {
        "products": products,
        "isDemo": False,
        "shelfSummary": parsed.get("shelfSummary"),
    }