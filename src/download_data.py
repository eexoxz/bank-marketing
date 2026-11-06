"""Fetch the original UCI archive and copy only the two required data files."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from bank_data import DATA_FILE, RAW_SHA256, write_bytes, write_json

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip'
SOURCE_PAGE = 'https://archive.ics.uci.edu/dataset/222/bank+marketing'


def extract_dataset(archive: bytes) -> dict:
    raw = ROOT / 'data' / 'raw'
    selected = {}
    payloads = {}
    with ZipFile(io.BytesIO(archive)) as outer:
        candidates = [name for name in outer.namelist() if name.endswith('bank-additional.zip')]
        if len(candidates) != 1:
            raise ValueError('Expected one bank-additional.zip inside the UCI archive.')
        with ZipFile(io.BytesIO(outer.read(candidates[0]))) as inner:
            for filename in (DATA_FILE, 'bank-additional-names.txt'):
                matches = [name for name in inner.namelist() if name.endswith('/' + filename) or name == filename]
                if len(matches) != 1:
                    raise ValueError(f'Expected one {filename} in the additional archive.')
                payload = inner.read(matches[0])
                payloads[filename] = payload
                selected[filename] = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    if selected[DATA_FILE]['sha256'] != RAW_SHA256:
        raise ValueError('The archive does not contain the canonical coursework CSV.')
    manifest = {
        'source_page': SOURCE_PAGE,
        'archive_url': URL,
        'prepared_on_local_date': '2026-10-03',
        'archive_sha256': hashlib.sha256(archive).hexdigest(),
        'variant': DATA_FILE,
        'licence': 'Creative Commons Attribution 4.0',
        'citation': 'Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306',
        'files': selected,
    }
    raw.mkdir(parents=True, exist_ok=True)
    for filename, payload in payloads.items():
        write_bytes(raw / filename, payload)
    write_json(raw / 'source_manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded archive.')
    args = parser.parse_args()
    if args.archive:
        archive = args.archive.read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            archive = response.read()
    print(json.dumps(extract_dataset(archive), indent=2))
