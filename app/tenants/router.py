from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.tenants.dependencies import CurrentTenant
from app.tenants.schemas import TenantCreate, TenantCreated, TenantRead
from app.tenants.service import create_tenant

router = APIRouter(prefix="/tenants", tags=["tenants"])


@router.post("", response_model=TenantCreated, status_code=status.HTTP_201_CREATED)
async def create(
    payload: TenantCreate,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> TenantCreated:
    tenant, api_key = await create_tenant(session, payload.name)
    return TenantCreated(
        id=tenant.id,
        name=tenant.name,
        created_at=tenant.created_at,
        api_key=api_key,
    )


@router.get("/me", response_model=TenantRead)
async def read_me(tenant: CurrentTenant) -> TenantRead:
    return TenantRead.model_validate(tenant)
