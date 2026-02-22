"""
Configuration — reads from environment variables.
When OPENAI_API_KEY is missing the whole app runs in DEMO mode
so the frontend can still be tested end-to-end.
"""
from dotenv import load_dotenv
load_dotenv()

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# S3 (optional — images work without it, stored temporarily on disk)
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

# CORS — frontend Amplify + local dev
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,https://main.d22y6a7qhuis5g.amplifyapp.com,https://healthennutrition.com.mx,https://www.healthennutrition.com.mx,https://nutriscan.healthennutrition.com.mx"
).split(",")

IS_DEMO = not bool(OPENAI_API_KEY)
