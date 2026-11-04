from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from openpyxl import load_workbook

LABELS = ("c1", "c234", "c5")


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def group(text: str) -> str:
    return digest(clean(text).lower())


@dataclass(frozen=True)
class Text:
    excel_row: int
    text: str

    @property
    def text_sha256(self) -> str:
        return digest(self.text)


@dataclass(frozen=True)
class Example:
    excel_row: int
    text: str
    label: str


def read_xlsx(path: Path, *, labeled: bool = False):
    """Inference returns Text objects without labels, even for a labeled workbook."""
    if path.suffix.lower() != ".xlsx":
        raise ValueError("A entrada deve ser .xlsx.")
    workbook = load_workbook(path, read_only=True, data_only=False)
    records, coerced = [], []
    try:
        sheet = workbook.worksheets[0]
        rows = sheet.iter_rows()
        headers = [c.value for c in next(rows, ())]
        if not headers or None in headers or len(set(headers)) != len(headers):
            raise ValueError("Cabeçalhos vazios ou duplicados.")
        if "resp_text" not in headers or (labeled and "clarity" not in headers):
            raise ValueError("Colunas obrigatórias: resp_text e, para rótulos, clarity.")
        ti = headers.index("resp_text")
        li = headers.index("clarity") if labeled else None
        for row_number, cells in enumerate(rows, 2):
            cell = cells[ti]
            if cell.data_type == "f":
                raise ValueError(f"Fórmula em resp_text, linha {row_number}.")
            value = cell.value
            if value is not None and not isinstance(value, str):
                coerced.append(row_number)
            text = "" if value is None else str(value)
            if labeled:
                label = cells[li].value
                if cells[li].data_type == "f" or label not in LABELS:
                    raise ValueError(f"Rótulo inválido na linha {row_number}.")
                records.append(Example(row_number, text, label))
            else:
                records.append(Text(row_number, text))
        if not records:
            raise ValueError("Planilha sem registros.")
        return records, {"rows": len(records), "sheet": sheet.title, "columns": headers,
                         "coerced_text_excel_rows": coerced, "sha256": file_digest(path)}
    finally:
        workbook.close()


def describe(path: Path, *, labeled: bool = False) -> dict:
    records, info = read_xlsx(path, labeled=labeled)
    groups = defaultdict(list)
    for r in records:
        groups[group(r.text)].append(r)
    info.update(unique_normalized_texts=len(groups),
                empty_texts=sum(not clean(r.text) for r in records),
                max_characters=max(len(r.text) for r in records))
    if labeled:
        counts = Counter(r.label for r in records)
        info.update(class_counts=dict(counts), majority_accuracy=max(counts.values()) / len(records),
                    conflicting_groups=sum(len({r.label for r in rs}) > 1 for rs in groups.values()))
    return info


def atomic_text(path: Path, text: str, *, replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(name, path)
        else:
            os.link(name, path)  # Atomic exclusive creation; never overwrite a deliverable.
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value, *, replace: bool = True) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", replace=replace)


def write_jsonl(path: Path, rows) -> None:
    atomic_text(path, "".join(canonical(asdict(r) if hasattr(r, "__dataclass_fields__") else r) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSONL deve conter objetos e ao menos um registro.")
    return rows
