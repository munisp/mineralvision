"""
Local Cryptographic Provenance Ledger for MineralVision.

A REAL, self-contained cryptographic ledger — no simulated hashes, no fake
transactions. Every block is signed and the whole chain is verifiable.

Signing scheme
--------------
- **Ed25519** (via the ``cryptography`` package) when importable — the
  default and recommended scheme. Private key comes from the
  ``MV_LEDGER_PRIVATE_KEY`` env var (hex-encoded 32-byte seed); when unset,
  an ephemeral key is generated with a loud warning (dev only — the chain
  is not verifiable across restarts with a new key).
- **HMAC-SHA256** fallback (same pattern as the custody_ledger module)
  when ``cryptography`` is not installed, keyed by ``MV_LEDGER_HMAC_KEY``
  (hex) or an ephemeral random key. This is symmetric — verification
  requires the key — so ed25519 is strongly preferred.

Block structure
---------------
Each block contains: ``index``, ``timestamp`` (UTC ISO-8601),
``prev_hash``, ``merkle_root`` (SHA-256 Merkle tree over canonical
transaction payloads), ``tx_count``, ``transactions``, a ``signature``
over the canonical header, and ``hash`` = SHA-256(canonical header +
signature). ``verify_chain()`` walks the chain re-checking every link,
Merkle root, and signature.
"""

import hashlib
import hmac as hmac_lib
import json
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    ED25519_AVAILABLE = True
except ImportError:
    ED25519_AVAILABLE = False
    logger.warning(
        "cryptography package not installed — ledger falls back to "
        "HMAC-SHA256 signatures (symmetric). Install cryptography for "
        "ed25519 asymmetric signing."
    )

LEDGER_PRIVATE_KEY_ENV = "MV_LEDGER_PRIVATE_KEY"
LEDGER_HMAC_KEY_ENV = "MV_LEDGER_HMAC_KEY"


def _canonical(obj: Any) -> bytes:
    """Canonical JSON encoding for hashing/signing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def merkle_root(leaves: List[bytes]) -> str:
    """Compute the SHA-256 Merkle root of payload leaves."""
    if not leaves:
        return _sha256_hex(b"")
    level = [_sha256_hex(leaf) for leaf in leaves]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [
            _sha256_hex(bytes.fromhex(level[i]) + bytes.fromhex(level[i + 1]))
            for i in range(0, len(level), 2)
        ]
    return level[0]


class LedgerSigner:
    """ed25519 signer with HMAC-SHA256 fallback."""

    def __init__(self):
        self.scheme = "ed25519" if ED25519_AVAILABLE else "hmac-sha256"
        self._private_key: Optional[Ed25519PrivateKey] = None
        self._hmac_key: Optional[bytes] = None

        if ED25519_AVAILABLE:
            seed_hex = os.environ.get(LEDGER_PRIVATE_KEY_ENV)
            if seed_hex:
                self._private_key = Ed25519PrivateKey.from_private_bytes(
                    bytes.fromhex(seed_hex)
                )
            else:
                self._private_key = Ed25519PrivateKey.generate()
                logger.warning(
                    "%s is not set — generated an EPHEMERAL ed25519 ledger "
                    "key (dev only). Chains signed now will not verify "
                    "against a different key after restart.",
                    LEDGER_PRIVATE_KEY_ENV,
                )
        else:
            key_hex = os.environ.get(LEDGER_HMAC_KEY_ENV)
            if key_hex:
                self._hmac_key = bytes.fromhex(key_hex)
            else:
                self._hmac_key = secrets.token_bytes(32)
                logger.warning(
                    "%s is not set — generated an EPHEMERAL HMAC ledger "
                    "key (dev only).",
                    LEDGER_HMAC_KEY_ENV,
                )

    def sign(self, message: bytes) -> str:
        """Sign a message; returns hex signature."""
        if self._private_key is not None:
            return self._private_key.sign(message).hex()
        return hmac_lib.new(self._hmac_key, message, hashlib.sha256).hexdigest()

    def verify(self, message: bytes, signature_hex: str) -> bool:
        """Verify a signature."""
        try:
            if self._private_key is not None:
                self.public_key().verify(bytes.fromhex(signature_hex), message)
                return True
            expected = hmac_lib.new(
                self._hmac_key, message, hashlib.sha256
            ).hexdigest()
            return secrets.compare_digest(expected, signature_hex)
        except Exception:
            return False

    def public_key(self) -> Optional[Ed25519PublicKey]:
        return self._private_key.public_key() if self._private_key else None

    def public_key_hex(self) -> Optional[str]:
        pk = self.public_key()
        if pk is None:
            return None
        return pk.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        ).hex()

    @property
    def address(self) -> str:
        """Ledger account address derived from the public/verification key."""
        if self._private_key is not None:
            material = bytes.fromhex(self.public_key_hex())
        else:
            # HMAC is symmetric; derive a stable address from the key hash
            material = hashlib.sha256(self._hmac_key).digest()
        return "0x" + _sha256_hex(material)[:40]


@dataclass
class Block:
    """A signed ledger block."""
    index: int
    timestamp: str
    prev_hash: str
    merkle_root: str
    tx_count: int
    transactions: List[Dict[str, Any]]
    signature: str
    hash: str = ""

    def header(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "prev_hash": self.prev_hash,
            "merkle_root": self.merkle_root,
            "tx_count": self.tx_count,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.header(),
            "transactions": self.transactions,
            "signature": self.signature,
            "hash": self.hash,
        }


class LocalCryptoLedger:
    """
    Real local cryptographic ledger: signed, Merkle-anchored, verifiable.
    """

    CONTRACT_ADDRESS = "0x" + _sha256_hex(b"MineralProvenanceContract")[:40]

    def __init__(self, signer: LedgerSigner = None):
        self.signer = signer or LedgerSigner()
        self.scheme = self.signer.scheme
        self.blocks: List[Block] = []
        self._create_genesis()

    def _create_genesis(self) -> None:
        """Create the signed genesis block."""
        genesis_tx = {
            "type": "genesis",
            "mineralvision_ledger": True,
            "scheme": self.scheme,
            "address": self.signer.address,
        }
        block = self._build_block(0, "0" * 64, [genesis_tx])
        self.blocks.append(block)

    def _build_block(self, index: int, prev_hash: str,
                     transactions: List[Dict[str, Any]]) -> Block:
        leaves = [_canonical(tx) for tx in transactions]
        block = Block(
            index=index,
            timestamp=datetime.utcnow().isoformat(),
            prev_hash=prev_hash,
            merkle_root=merkle_root(leaves),
            tx_count=len(transactions),
            transactions=list(transactions),
            signature="",
        )
        block.signature = self.signer.sign(_canonical(block.header()))
        block.hash = _sha256_hex(_canonical(block.header()) + block.signature.encode())
        return block

    def add_block(self, transactions: List[Dict[str, Any]]) -> Block:
        """Append a new signed block of transactions to the chain."""
        if not transactions:
            raise ValueError("A block requires at least one transaction")
        block = self._build_block(
            len(self.blocks), self.blocks[-1].hash, transactions
        )
        self.blocks.append(block)
        return block

    def transaction_hash(self, tx: Dict[str, Any]) -> str:
        """Real content hash of a transaction payload."""
        return "0x" + _sha256_hex(_canonical(tx))

    def verify_chain(self) -> Dict[str, Any]:
        """
        Walk the whole chain verifying prev links, Merkle roots and
        signatures. Returns a verification report.
        """
        errors: List[str] = []

        for i, block in enumerate(self.blocks):
            if block.index != i:
                errors.append(f"block {i}: index mismatch ({block.index})")

            # Recompute hash
            expected_hash = _sha256_hex(
                _canonical(block.header()) + block.signature.encode()
            )
            if block.hash != expected_hash:
                errors.append(f"block {i}: hash mismatch")

            # Verify signature
            if not self.signer.verify(_canonical(block.header()), block.signature):
                errors.append(f"block {i}: INVALID SIGNATURE")

            # Verify prev link
            if i > 0 and block.prev_hash != self.blocks[i - 1].hash:
                errors.append(f"block {i}: prev_hash does not match block {i - 1}")

            # Verify Merkle root
            leaves = [_canonical(tx) for tx in block.transactions]
            if block.merkle_root != merkle_root(leaves):
                errors.append(f"block {i}: Merkle root mismatch")

            if block.tx_count != len(block.transactions):
                errors.append(f"block {i}: tx_count mismatch")

        return {
            "valid": not errors,
            "scheme": self.scheme,
            "height": len(self.blocks),
            "address": self.signer.address,
            "public_key": self.signer.public_key_hex(),
            "errors": errors,
            "verified_at": datetime.utcnow().isoformat(),
        }

    def get_block(self, index: int) -> Optional[Block]:
        if 0 <= index < len(self.blocks):
            return self.blocks[index]
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scheme": self.scheme,
            "height": len(self.blocks),
            "address": self.signer.address,
            "blocks": [b.to_dict() for b in self.blocks],
        }
