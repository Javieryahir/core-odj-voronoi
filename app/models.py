"""
Pydantic models — these match the frontend's expected JSON shapes.
Fields that the LLM can't determine from the image are Optional
so partial data is handled gracefully.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class BoundingBox(BaseModel):
    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0


class Product(BaseModel):
    id: str
    name: str
    brand: Optional[str] = "Desconocido"
    category: Optional[str] = "General"
    healthScore: int = Field(ge=0, le=100, default=50)

    # Nutritional — all Optional because the LLM may not know every value
    calories: Optional[float] = None
    sugar_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    fat_g: Optional[float] = None
    saturated_fat_g: Optional[float] = None
    fiber_g: Optional[float] = None
    protein_g: Optional[float] = None

    # NOM-051 warning seals
    sellos: List[str] = []

    recommendation: Optional[str] = ""
    bbox: BoundingBox = BoundingBox()


class AnalyzeResponse(BaseModel):
    products: List[Product]
    isDemo: bool = False
    shelfSummary: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    products: Optional[List[Product]] = []


class ChatResponse(BaseModel):
    reply: str
