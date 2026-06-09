import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TenantCreate(BaseModel):
    """Input for creating a tenant."""

    name: str = Field(min_length=1, max_length=255)


class TenantRead(BaseModel):
    """Public view of a tenant. Never includes the API key."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    created_at: datetime


class TenantCreated(TenantRead):
    """Returned once on creation. Carries the plaintext API key, shown only here."""

    api_key: str
