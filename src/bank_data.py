"""Canonical data location and integrity checks for the coursework dataset."""
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


DATA_FILE = 'bank-additional-full.csv'
RAW_SHA256 = '74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8'
BOUNDARY = 32950
TOTAL = 41188


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source(root: Path) -> tuple[Path, str]:
    path = root / 'data' / 'raw' / DATA_FILE
    manifest_path = path.with_name('source_manifest.json')
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError('Raw data or source manifest missing. Run python src/download_data.py.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    digest = sha(path)
    if digest != RAW_SHA256 or digest != manifest['files'][DATA_FILE]['sha256']:
        raise ValueError('Raw data does not match the canonical Bank Marketing CSV and source manifest.')
    return path, digest


def write_bytes(path: Path, content: bytes) -> None:
    """Replace a file only after its new contents have been fully written."""
    temp = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', delete=False) as stream:
            temp = Path(stream.name)
            stream.write(content)
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    write_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8'))
