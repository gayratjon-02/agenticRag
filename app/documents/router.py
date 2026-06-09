import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.documents.schemas import DocumentRead
from app.documents.service import create_document, get_document
from app.documents.tasks import ingest_document_task
from app.security.rate_limit import rate_limit
from app.tenants.dependencies import CurrentTenant

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(rate_limit)])


@router.post("", response_model=DocumentRead, status_code=status.HTTP_202_ACCEPTED)
async def upload(
    tenant: CurrentTenant,
    session: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile,
) -> DocumentRead:
    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only UTF-8 text documents are supported",
        ) from exc
    if not content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is empty",
        )

    document = await create_document(session, tenant.id, file.filename or "untitled", content)
    # Heavy work runs in the background; the request returns immediately.
    ingest_document_task.delay(str(document.id))
    return DocumentRead.model_validate(document)


@router.get("/{document_id}", response_model=DocumentRead)
async def get_status(
    document_id: uuid.UUID,
    tenant: CurrentTenant,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentRead:
    document = await get_document(session, tenant.id, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentRead.model_validate(document)
