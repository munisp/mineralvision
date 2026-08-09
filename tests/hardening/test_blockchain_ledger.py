"""Tests for the real cryptographic provenance ledger (sign/verify/tamper)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "MineralVision_Final_Package"))

from src.api.blockchain.advanced_blockchain import (  # noqa: E402
    create_advanced_blockchain_manager,
)
from src.api.blockchain.local_ledger import (  # noqa: E402
    LedgerSigner,
    LocalCryptoLedger,
    merkle_root,
)


def test_ed25519_sign_and_verify():
    signer = LedgerSigner()
    assert signer.scheme == "ed25519"  # cryptography is installed in this env
    sig = signer.sign(b"hello provenance")
    assert signer.verify(b"hello provenance", sig) is True
    assert signer.verify(b"tampered message", sig) is False
    assert signer.address.startswith("0x") and len(signer.address) == 42


def test_ledger_genesis_and_append():
    ledger = LocalCryptoLedger()
    assert ledger.blocks[0].index == 0
    report = ledger.verify_chain()
    assert report["valid"] is True
    assert report["height"] == 1

    block = ledger.add_block([{"action": "register_data", "data_hash": "abc"}])
    assert block.index == 1
    assert block.prev_hash == ledger.blocks[0].hash
    assert block.merkle_root == merkle_root([b'{"action":"register_data","data_hash":"abc"}'])

    report = ledger.verify_chain()
    assert report["valid"] is True
    assert report["height"] == 2
    assert report["scheme"] == "ed25519"


def test_ledger_tamper_detection_payload():
    ledger = LocalCryptoLedger()
    ledger.add_block([{"action": "register_data", "data_hash": "original"}])
    assert ledger.verify_chain()["valid"] is True

    # Tamper with a recorded transaction
    ledger.blocks[1].transactions[0]["data_hash"] = "forged"
    report = ledger.verify_chain()
    assert report["valid"] is False
    assert any("Merkle root mismatch" in e for e in report["errors"])


def test_ledger_tamper_detection_signature():
    ledger = LocalCryptoLedger()
    ledger.add_block([{"action": "x"}])
    # Forge a signature
    ledger.blocks[1].signature = "00" * 64
    report = ledger.verify_chain()
    assert report["valid"] is False
    assert any("INVALID SIGNATURE" in e for e in report["errors"])


def test_ledger_tamper_detection_broken_link():
    ledger = LocalCryptoLedger()
    ledger.add_block([{"action": "a"}])
    ledger.add_block([{"action": "b"}])
    ledger.blocks[2].prev_hash = "ff" * 32
    report = ledger.verify_chain()
    assert report["valid"] is False
    assert any("prev_hash" in e for e in report["errors"])


def test_manager_register_data_anchored_and_verifiable():
    manager = create_advanced_blockchain_manager(signers=["s1", "s2"])
    result = manager.register_data(
        data_hash="hash-1", ipfs_hash="ipfs-1",
        data_type="assay", metadata={"project": "p1"},
    )

    tx = result["transaction"]
    assert tx["tx_hash"].startswith("0x")
    assert tx["block_number"] == 1  # real ledger height, not 12345678
    assert tx["from_address"] == manager.ledger.signer.address
    assert tx["to_address"] == manager.ledger.CONTRACT_ADDRESS

    assert result["ledger"]["scheme"] == "ed25519"
    assert result["ledger"]["block_index"] == 1
    assert result["ledger"]["signature"]

    report = manager.verify_ledger()
    assert report["valid"] is True
    assert report["height"] == 2

    # No simulated constants anywhere
    assert tx["from_address"] != "0x" + "0" * 40
    assert tx["to_address"] != "0x" + "1" * 40


def test_manager_multisig_execution_recorded():
    manager = create_advanced_blockchain_manager(signers=["s1", "s2"])
    result = manager.register_data(
        "h", "i", "drillhole", {}, use_multisig=True,
    )
    proposal_id = result["proposal"]["proposal_id"]
    # The proposer's signature is already counted; approve with the other signer
    already = set(result["proposal"]["signatures"])
    for signer in ("s1", "s2"):
        if signer not in already:
            manager.multisig_manager.approve(proposal_id, signer)
    manager.multisig_manager.execute(proposal_id)

    proposal = manager.multisig_manager.get_proposal(proposal_id)
    rec = manager.record_multisig_execution(proposal)
    assert rec["block_index"] >= 1
    assert manager.verify_ledger()["valid"] is True
