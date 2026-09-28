from __future__ import annotations

from pathlib import Path

from neurofence.acquisition.hashing import hash_file


def test_hash_is_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello world" * 1000)

    h1 = hash_file(f)
    h2 = hash_file(f)

    assert h1 == h2


def test_single_byte_change_changes_hash(tmp_path: Path) -> None:
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    data = bytearray(b"a" * 10_000)
    f1.write_bytes(bytes(data))

    data[5000] ^= 0x01
    f2.write_bytes(bytes(data))

    sha256_a, sha512_a = hash_file(f1)
    sha256_b, sha512_b = hash_file(f2)

    assert sha256_a != sha256_b
    assert sha512_a != sha512_b


def test_empty_file_hashes_to_known_constant(tmp_path: Path) -> None:
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")

    sha256, sha512 = hash_file(f)

    expected_sha512 = (
        "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0"
        "ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
    )
    assert sha256 == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert sha512 == expected_sha512


def test_large_file_streaming(tmp_path: Path) -> None:
    """Hashing must not require loading the whole file into memory at once;
    exercise a file larger than the internal chunk size."""
    f = tmp_path / "large.bin"
    chunk = b"x" * (1024 * 1024)
    with open(f, "wb") as fh:
        for _ in range(6):  # 6 MiB, > 4 MiB chunk size
            fh.write(chunk)

    sha256, sha512 = hash_file(f)
    assert len(sha256) == 64
    assert len(sha512) == 128
