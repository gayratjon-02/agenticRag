from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.tenants.models import Tenant
from app.tenants.security import generate_api_key, hash_api_key


async def create_tenant(session: AsyncSession, name: str) -> tuple[Tenant, str]:
    """Create a tenant and return it together with its plaintext API key.

    The plaintext key is returned only here; only its hash is stored.
    """
    api_key = generate_api_key()
    tenant = Tenant(name=name, api_key_hash=hash_api_key(api_key))
    session.add(tenant)
    await session.commit()
    await session.refresh(tenant)
    return tenant, api_key


async def get_tenant_by_api_key(session: AsyncSession, api_key: str) -> Tenant | None:
    """Resolve a tenant from a plaintext API key, or None if no match."""
    key_hash = hash_api_key(api_key)
    result = await session.execute(select(Tenant).where(Tenant.api_key_hash == key_hash))
    return result.scalars().one_or_none()
