import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.documents.models import IngestionStatus


class DocumentRead(BaseModel):
    """Status view of an uploaded document."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source: str
    status: IngestionStatus
    chunk_count: int
    error: str | None
    created_at: datetime
