from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from app.services.keys_repo import verify_key

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(api_key_header)) -> str:
    if api_key is None or not verify_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key
