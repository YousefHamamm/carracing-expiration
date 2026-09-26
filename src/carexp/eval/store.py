"""Per-cell result files with resume.

<root>/meta.json         what the results depend on (checkpoint hash, tracks, env, channel seed, ...);
                         a restart with different meta is refused so cells are never mixed
<root>/cells/<key>.csv   one row per episode, written atomically when the cell finishes
<root>/cells/<key>.json  the cell's summary
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


def file_sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", newline="") as f:
        write(f)
    os.replace(tmp, path)


class CellStore:
    def __init__(self, root, meta: dict):
        self.root = Path(root)
        self.cells = self.root / "cells"
        self.cells.mkdir(parents=True, exist_ok=True)
        meta_path = self.root / "meta.json"
        meta = json.loads(json.dumps(meta))  # normalize tuples etc. for comparison
        if meta_path.exists():
            old = json.loads(meta_path.read_text())
            if old != meta:
                diff = sorted(k for k in set(old) | set(meta) if old.get(k) != meta.get(k))
                raise ValueError(f"{meta_path} was written for a different setup (differs in: {diff}); "
                                 "use a new --out-dir")
        else:
            _atomic_write(meta_path, lambda f: json.dump(meta, f, indent=2))
        self.meta = meta

    def _csv(self, key: str) -> Path:
        return self.cells / f"{key}.csv"

    def is_done(self, key: str, n_expected: int) -> bool:
        path = self._csv(key)
        if not path.exists() or not (self.cells / f"{key}.json").exists():
            return False
        return len(self.load_rows(key)) == n_expected

    def load_rows(self, key: str) -> list[dict]:
        with open(self._csv(key), newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["seed"] = int(r["seed"])
            r["ret"] = float(r["ret"])
            r["track_fraction"] = float(r["track_fraction"])
            r["lap_finished"] = r["lap_finished"] == "True"
            for k in ("length", "transmissions", "deliveries"):
                r[k] = int(r[k])
        return rows

    def load_summary(self, key: str) -> dict:
        return json.loads((self.cells / f"{key}.json").read_text())

    def save(self, key: str, rows: list[dict], summary: dict) -> None:
        def write_csv(f):
            w = csv.DictWriter(f, list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        _atomic_write(self._csv(key), write_csv)
        _atomic_write(self.cells / f"{key}.json", lambda f: json.dump(summary, f, indent=2))
