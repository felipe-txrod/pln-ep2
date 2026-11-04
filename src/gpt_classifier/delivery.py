from __future__ import annotations

import os
import posixpath
import tempfile
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from openpyxl import load_workbook

from .data import LABELS, digest, read_xlsx

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def export_test2(source: Path, target: Path, predictions: list[dict]) -> None:
    """Official delivery: exactly the 897 unlabeled test2 records."""
    _export_copy(source, target, predictions, expected_rows=897, dataset="teste2")


def export_test1(source: Path, target: Path, predictions: list[dict]) -> None:
    """Comparison artifact only; never the official EP2 submission."""
    _export_copy(source, target, predictions, expected_rows=900, dataset="teste1 (comparação)")


def _export_copy(source: Path, target: Path, predictions: list[dict], *, expected_rows: int, dataset: str) -> None:
    """Change only blank clarity cell values in the original OOXML archive.

    All other ZIP entries are copied verbatim, including styles, other sheets,
    relationships and workbook settings. No recreation/reformatting of the sheet.
    """
    if target.suffix.lower() != ".xlsx" or source.resolve() == target.resolve() or target.exists():
        raise ValueError("Use uma nova saída .xlsx diferente da entrada.")
    texts, info = read_xlsx(source)
    if len(texts) != expected_rows or info["columns"] != ["resp_text", "clarity"]:
        raise ValueError(f"Use o modelo de {dataset} com {expected_rows} linhas e duas colunas originais.")
    if len(predictions) != len(texts):
        raise ValueError(f"Quantidade incorreta de previsões para {dataset}.")
    for text, prediction in zip(texts, predictions):
        if (prediction.get("excel_row") != text.excel_row
                or prediction.get("text_sha256") != digest(text.text) or prediction.get("label") not in LABELS):
            raise ValueError("Previsões inválidas/desalinhadas para exportação.")
    workbook = load_workbook(source, read_only=True)
    try:
        if any(row[0] is not None for row in workbook.worksheets[0].iter_rows(min_row=2, min_col=2, max_col=2, values_only=True)):
            raise ValueError(f"A entrada de {dataset} deve ter clarity vazia; não use o gabarito na exportação.")
    finally:
        workbook.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".xlsx")
    os.close(fd)
    try:
        with ZipFile(source) as original:
            wb = ET.fromstring(original.read("xl/workbook.xml"))
            first = wb.find(f"{{{NS}}}sheets")[0]
            rid = first.attrib[f"{{{REL_NS}}}id"]
            relationships = ET.fromstring(original.read("xl/_rels/workbook.xml.rels"))
            relation = next(r for r in relationships if r.attrib["Id"] == rid)
            sheet_target = relation.attrib["Target"]
            name = sheet_target.lstrip("/") if sheet_target.startswith("/") else posixpath.normpath("xl/" + sheet_target)
            sheet_xml = original.read(name)
            namespaces = dict(value for _, value in ET.iterparse(BytesIO(sheet_xml), events=("start-ns",)))
            for prefix, uri in namespaces.items():
                ET.register_namespace(prefix, uri)
            root = ET.fromstring(sheet_xml)
            sheet_data = root.find(f"{{{NS}}}sheetData")
            rows = {int(row.attrib["r"]): row for row in sheet_data}
            for prediction in predictions:
                number = prediction["excel_row"]
                row = rows[number]
                coordinate = f"B{number}"
                cell = next((c for c in row if c.attrib.get("r") == coordinate), None)
                if cell is None:
                    cell = ET.SubElement(row, f"{{{NS}}}c", {"r": coordinate})
                for child in list(cell):
                    cell.remove(child)
                cell.set("t", "inlineStr")
                text_node = ET.SubElement(ET.SubElement(cell, f"{{{NS}}}is"), f"{{{NS}}}t")
                text_node.text = prediction["label"]
            xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            # Keep declarations referenced lexically by mc:Ignorable, even when
            # the namespace has no element/attribute and ElementTree drops it.
            for prefix, uri in namespaces.items():
                declaration = f"xmlns:{prefix}" if prefix else "xmlns"
                if (declaration + "=").encode() not in xml:
                    root.set(declaration, uri)
            xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            with ZipFile(temporary, "w") as result:
                result.comment = original.comment
                for entry in original.infolist():
                    result.writestr(entry, xml if entry.filename == name else original.read(entry.filename))
        # Reopen independently before publishing a new deliverable.
        checked, checked_info = read_xlsx(Path(temporary), labeled=True)
        if checked_info["columns"] != info["columns"] or len(checked) != expected_rows:
            raise ValueError("Formato da entrega alterado.")
        for text, saved, prediction in zip(texts, checked, predictions):
            if saved.text != text.text or saved.label != prediction["label"]:
                raise ValueError("Falha na verificação da entrega.")
        os.link(temporary, target)  # Exclusive publication, including concurrent writers.
    finally:
        os.unlink(temporary)
