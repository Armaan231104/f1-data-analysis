from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime

from sqlalchemy import select, update

from app.core.config import settings
from app.core.db import api_keys, get_engine


def _hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def add_key(label: str) -> str:
    """Generate a new API key, store its hash, and return the plaintext once."""
    plaintext = secrets.token_urlsafe(32)
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            api_keys.insert().values(
                key_hash=_hash_key(plaintext), label=label, created_at=_now(), revoked_at=None
            )
        )
    return plaintext


def revoke_key(key_hash: str) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            update(api_keys).where(api_keys.c.key_hash == key_hash).values(revoked_at=_now())
        )


def verify_key(plaintext: str) -> bool:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            select(api_keys.c.id).where(
                api_keys.c.key_hash == _hash_key(plaintext), api_keys.c.revoked_at.is_(None)
            )
        ).first()
    return row is not None


def seed_from_env() -> None:
    """Seed bootstrap keys from settings.api_keys_seed on first boot, if the table is empty."""
    if not settings.api_keys_seed:
        return
    engine = get_engine()
    with engine.connect() as conn:
        if conn.execute(select(api_keys.c.id).limit(1)).first() is not None:
            return
    with engine.begin() as conn:
        for raw_key in settings.api_keys_seed.split(","):
            key = raw_key.strip()
            if not key:
                continue
            conn.execute(
                api_keys.insert().values(
                    key_hash=_hash_key(key), label="env-seed", created_at=_now(), revoked_at=None
                )
            )
