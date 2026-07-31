from fastapi import Header, HTTPException, status

from config.settings import settings


async def verify_internal_api_key(x_internal_api_key: str | None = Header(default=None)) -> None:
    if not settings.internal_api_key or x_internal_api_key != settings.internal_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Clé API interne invalide ou manquante.")
