"""
S3 storage — uploads shelf images for logging/history.
Completely optional: when AWS_S3_BUCKET is empty, images are
processed in-memory and not persisted.
"""

import uuid
from datetime import datetime, timezone
from app.config import AWS_S3_BUCKET, AWS_REGION


async def upload_image(image_bytes: bytes, content_type: str = "image/jpeg") -> str | None:
    """
    Upload image to S3 and return the URL.
    Returns None if S3 is not configured (image still gets analyzed).
    """
    if not AWS_S3_BUCKET:
        return None

    try:
        import boto3

        s3 = boto3.client("s3", region_name=AWS_REGION)
        timestamp = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        key = f"scans/{timestamp}/{uuid.uuid4().hex}.jpg"

        s3.put_object(
            Bucket=AWS_S3_BUCKET,
            Key=key,
            Body=image_bytes,
            ContentType=content_type,
        )

        return f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"

    except Exception as e:
        print(f"[S3] Upload failed (non-blocking): {e}")
        return None