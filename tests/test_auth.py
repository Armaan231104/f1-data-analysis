"""Tests for the API key store: add/revoke/verify and env-seeding."""

from __future__ import annotations

from app.core import config as config_module
from app.services import keys_repo


def test_add_key_then_verify(tmp_db) -> None:
    plaintext = keys_repo.add_key("test-client")
    assert keys_repo.verify_key(plaintext) is True


def test_unknown_key_does_not_verify(tmp_db) -> None:
    assert keys_repo.verify_key("not-a-real-key") is False


def test_revoked_key_no_longer_verifies(tmp_db) -> None:
    plaintext = keys_repo.add_key("test-client")
    assert keys_repo.verify_key(plaintext) is True

    key_hash = keys_repo._hash_key(plaintext)
    keys_repo.revoke_key(key_hash)

    assert keys_repo.verify_key(plaintext) is False


def test_seed_from_env_creates_keys(tmp_db, monkeypatch) -> None:
    monkeypatch.setattr(config_module.settings, "api_keys_seed", "seed-key-1,seed-key-2")
    keys_repo.seed_from_env()

    assert keys_repo.verify_key("seed-key-1") is True
    assert keys_repo.verify_key("seed-key-2") is True


def test_seed_from_env_is_idempotent(tmp_db, monkeypatch) -> None:
    monkeypatch.setattr(config_module.settings, "api_keys_seed", "seed-key-1")
    keys_repo.seed_from_env()
    keys_repo.seed_from_env()  # table is non-empty now -> should be a no-op, not a duplicate insert

    assert keys_repo.verify_key("seed-key-1") is True


def test_seed_from_env_noop_when_unset(tmp_db, monkeypatch) -> None:
    monkeypatch.setattr(config_module.settings, "api_keys_seed", "")
    keys_repo.seed_from_env()  # should not raise, should not seed anything

    assert keys_repo.verify_key("anything") is False
