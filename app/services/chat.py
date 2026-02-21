"""
Chat service — answers user questions about the detected products.
Receives the product context from the last analysis so GPT can
give specific, grounded answers about what's on the shelf.
"""

import json
import httpx
from typing import List, Optional
from app.config import OPENAI_API_KEY, OPENAI_MODEL, IS_DEMO
from app.models import Product
from app.services.demo_data import DEMO_CHAT_RESPONSES

CHAT_SYSTEM_PROMPT = """Eres NutriScan, un asistente de nutrición mexicano experto en productos de tiendas de conveniencia.

El usuario acaba de escanear un estante de tienda y estos son los productos detectados:

{product_context}

REGLAS:
- Responde SIEMPRE en español mexicano, informal pero informativo.
- Basa tus respuestas SOLO en los productos listados arriba.
- Usa emojis de semáforo: 🟢 saludable, 🟡 moderado, 🟠 poco saludable, 🔴 evitar.
- Cuando compares productos, incluye datos numéricos específicos.
- Si te preguntan por un nutriente, ordena los productos de mejor a peor.
- Da recomendaciones prácticas y directas.
- Si un dato nutricional es null/desconocido, dilo honestamente.
- Usa **negritas** para nombres de productos y datos importantes.
- Mantén las respuestas concisas (máximo 200 palabras).
- Referencia los sellos NOM-051 cuando sea relevante."""


def _build_product_context(products: List[dict]) -> str:
    """Format products into a readable context string for the LLM."""
    lines = []
    for p in products:
        sellos = ", ".join(p.get("sellos", [])) or "Ninguno"
        line = (
            f"- {p['name']} (Score: {p.get('healthScore', '?')}/100) | "
            f"Cal: {p.get('calories', '?')} | Azúcar: {p.get('sugar_g', '?')}g | "
            f"Sodio: {p.get('sodium_mg', '?')}mg | Proteína: {p.get('protein_g', '?')}g | "
            f"Sellos: {sellos}"
        )
        lines.append(line)
    return "\n".join(lines)


async def chat_with_context(
    message: str,
    products: Optional[List[dict]] = None,
) -> str:
    """
    Send user question + product context to GPT.
    Falls back to keyword-matched demo responses when no API key.
    """
    if IS_DEMO:
        return _demo_response(message)

    product_context = _build_product_context(products or [])
    system = CHAT_SYSTEM_PROMPT.replace("{product_context}", product_context)

    payload = {
        "model": OPENAI_MODEL,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
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
    return data["choices"][0]["message"]["content"]


def _demo_response(message: str) -> str:
    """Simple keyword matching for demo mode."""
    lower = message.lower()
    if any(w in lower for w in ["sana", "mejor", "recomiend", "saludable"]):
        return DEMO_CHAT_RESPONSES["sana"]
    if any(w in lower for w in ["azúcar", "azucar", "dulce"]):
        return DEMO_CHAT_RESPONSES["azucar"]
    if any(w in lower for w in ["proteína", "proteina"]):
        return DEMO_CHAT_RESPONSES["proteina"]
    if any(w in lower for w in ["sodio", "sal"]):
        return DEMO_CHAT_RESPONSES["sodio"]
    return DEMO_CHAT_RESPONSES["default"]