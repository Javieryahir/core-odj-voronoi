"""
Chat service — answers user questions about the detected products.
Receives the product context from the last analysis so GPT can
give specific, grounded answers about what's on the shelf.
"""

import asyncio
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


# ── Batch image chat (asynchronous calls per batch of 10) ──

CHAT_IMAGES_SYSTEM_PROMPT = """Eres NutriScan, un asistente de nutrición mexicano experto en productos de tiendas.

Analiza las imágenes de productos/estantes que te envío y responde de forma informativa.
- Usa emojis de semáforo: 🟢 saludable, 🟡 moderado, 🟠 poco saludable, 🔴 evitar.
- Responde en español mexicano, informal pero informativo.
- Referencia sellos NOM-051 cuando apliquen.
- Mantén las respuestas concisas."""


def _ensure_data_url(b64: str) -> str:
    """Ensure base64 string has data URL prefix for OpenAI API."""
    if b64.startswith("data:"):
        return b64
    return f"data:image/jpeg;base64,{b64}"


async def _call_chat_with_images_batch(
    images: List[str],
    message: Optional[str],
    client: httpx.AsyncClient,
) -> str:
    """Single API call for a batch of images (max 10)."""
    if IS_DEMO:
        return f"[Demo] Análisis de {len(images)} imagen(es). Modo demo activo."

    content: List[dict] = []
    for b64 in images:
        url = _ensure_data_url(b64)
        content.append({
            "type": "image_url",
            "image_url": {"url": url, "detail": "high"},
        })
    prompt = message or "Analiza estas imágenes de productos y da un resumen nutricional."
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": OPENAI_MODEL,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": CHAT_IMAGES_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    }

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


async def chat_with_images(
    images: List[str],
    message: Optional[str] = None,
    batch_size: int = 10,
) -> List[str]:
    """
    Envía un arreglo de imágenes (base64) a la Chat API en batches de 10.
    Realiza llamadas asíncronas por cada batch y devuelve las respuestas.
    """
    if not images:
        return []

    batches: List[List[str]] = []
    for i in range(0, len(images), batch_size):
        batches.append(images[i : i + batch_size])

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [
            _call_chat_with_images_batch(batch, message, client)
            for batch in batches
        ]
        replies = await asyncio.gather(*tasks, return_exceptions=True)

    result: List[str] = []
    for i, r in enumerate(replies):
        if isinstance(r, Exception):
            result.append(f"Error en batch {i + 1}: {str(r)}")
        else:
            result.append(r)
    return result