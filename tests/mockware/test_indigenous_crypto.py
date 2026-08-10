"""
Indigenous knowledge crypto hard-requirement tests.

Without the ``cryptography`` package, encryption of culturally sensitive
data must FAIL HONESTLY (RuntimeError with remediation) — never silently
degrade to XOR. With ``cryptography`` installed, AES-256-GCM roundtrips
must work. The insecure XOR path is reachable only via the explicit
opt-in env var MV_INDIGENOUS_ALLOW_XOR_FALLBACK=true.
"""

import builtins
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "MineralVision_Final_Package", "src"))

from api.indigenous_knowledge.advanced_indigenous import (  # noqa: E402
    AES256EncryptionProvider,
    EncryptedContent,
)

SENSITIVE = b"sacred site coordinates: -23.4421, 133.8807 (restricted)"


@pytest.fixture()
def no_cryptography(monkeypatch):
    """Simulate an environment where `cryptography` is not installed."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("cryptography"):
            raise ImportError("No module named 'cryptography'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("MV_INDIGENOUS_ALLOW_XOR_FALLBACK", raising=False)


def test_encrypt_without_cryptography_raises_honest_error(no_cryptography):
    provider = AES256EncryptionProvider()
    with pytest.raises(RuntimeError) as excinfo:
        provider.encrypt(SENSITIVE)
    msg = str(excinfo.value)
    assert "cryptography" in msg
    assert "pip install cryptography" in msg  # remediation included


def test_decrypt_without_cryptography_raises_honest_error(no_cryptography):
    provider = AES256EncryptionProvider()
    # hand-built AES-256-GCM envelope (as if produced before the outage)
    envelope = EncryptedContent(
        content_id="c1",
        encrypted_data=b"\x00" * 16,
        encryption_method="AES-256-GCM",
        key_id="k1",
        iv=b"\x00" * 12,
    )
    with pytest.raises(RuntimeError):
        provider.decrypt(envelope)


def test_xor_fallback_content_refused_without_optin(no_cryptography):
    provider = AES256EncryptionProvider()
    envelope = EncryptedContent(
        content_id="c2",
        encrypted_data=b"garbage",
        encryption_method="XOR-FALLBACK",
        key_id="k2",
        iv=b"\x00" * 16,
    )
    with pytest.raises(RuntimeError) as excinfo:
        provider.decrypt(envelope)
    assert "XOR" in str(excinfo.value)


def test_xor_fallback_requires_explicit_optin(no_cryptography, monkeypatch):
    monkeypatch.setenv("MV_INDIGENOUS_ALLOW_XOR_FALLBACK", "true")
    provider = AES256EncryptionProvider()
    encrypted = provider.encrypt(SENSITIVE)
    assert encrypted.encryption_method == "XOR-FALLBACK"
    assert provider.decrypt(encrypted) == SENSITIVE  # roundtrip under opt-in


def test_aes_gcm_roundtrip_with_cryptography():
    provider = AES256EncryptionProvider()
    encrypted = provider.encrypt(SENSITIVE)
    assert encrypted.encryption_method == "AES-256-GCM"
    assert encrypted.encrypted_data != SENSITIVE  # actually transformed
    assert len(encrypted.iv) == 12  # 96-bit GCM nonce
    assert provider.decrypt(encrypted) == SENSITIVE

    # tampered ciphertext must fail authentication (GCM tag)
    tampered = EncryptedContent(
        content_id=encrypted.content_id,
        encrypted_data=bytes([encrypted.encrypted_data[0] ^ 0xFF])
        + encrypted.encrypted_data[1:],
        encryption_method=encrypted.encryption_method,
        key_id=encrypted.key_id,
        iv=encrypted.iv,
    )
    with pytest.raises(Exception):
        provider.decrypt(tampered)
