from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.session import get_db
from app.tenants.models import Tenant
from app.tenants.service import get_tenant_by_api_key


async def get_current_tenant(
    session: Annotated[AsyncSession, Depends(get_db)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> Tenant:
    """Resolve the tenant for the request from the X-API-Key header.

    A missing or invalid key is rejected — there is never a default tenant.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )
    tenant = await get_tenant_by_api_key(session, x_api_key)
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return tenant


CurrentTenant = Annotated[Tenant, Depends(get_current_tenant)]


async def require_admin(
    settings: Annotated[Settings, Depends(get_settings)],
    x_admin_key: Annotated[str | None, Header()] = None,
) -> None:
    """Guard privileged endpoints with an admin key.

    If no admin key is configured (admin_api_key is None), the guard is open — this
    is intended for local dev only; production sets ADMIN_API_KEY.
    """
    if settings.admin_api_key is None:
        return
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin key required",
        )
