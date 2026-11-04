from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest
from scipy.stats import binomtest

from gpt_classifier.data import Example, digest, file_digest, read_xlsx
from gpt_classifier.delivery import export_test1, export_test2
from gpt_classifier.evaluation import align, compare

ROOT = Path(__file__).resolve().parents[1]


def pred(row, text, label):
    return {"excel_row": row, "text_sha256": digest(text), "label": label}


def test_mcnemar_counts_direction_exact_and_ties():
    gold = [Example(i + 2, f"text {i}", "c1") for i in range(900)]
    old = [Example(r.excel_row, r.text, "c1" if i < 450 else "c5") for i, r in enumerate(gold)]
    predictions = [pred(r.excel_row, r.text, "c1" if i < 600 else "c5") for i, r in enumerate(gold)]
    result = compare(gold, predictions, old, gold)
    assert result["mcnemar"]["contingency_rows_ep1_correct_incorrect_columns_ep2_correct_incorrect"] == [[450, 0], [150, 300]]
    assert result["mcnemar"]["p_exact_two_sided"] == binomtest(0, 150, 0.5).pvalue
    assert result["mcnemar"]["significantly_superior"] is True
    worse = [pred(r.excel_row, r.text, "c1" if i < 300 else "c5") for i, r in enumerate(gold)]
    assert compare(gold, worse, old, gold)["mcnemar"]["significantly_superior"] is False
    same = [pred(r.excel_row, r.text, r.label) for r in old]
    assert compare(gold, same, old, gold)["mcnemar"]["p_exact_two_sided"] == 1


def test_rejects_missing_reordered_or_wrong_text():
    gold = [Example(2, "a", "c1"), Example(3, "b", "c5")]
    rows = [pred(2, "a", "c1"), pred(3, "b", "c5")]
    for bad in (rows[:1], list(reversed(rows)), [rows[0], rows[0]], [rows[0], pred(3, "c", "c5")]):
        with pytest.raises(ValueError):
            align(gold, bad)
    with pytest.raises(ValueError, match="900"):
        compare(gold, rows, gold, gold)


def test_official_ep1_reference_is_432_of_900():
    gold, _ = read_xlsx(ROOT / "data/test1-rotulado-para-teste-estatistico.xlsx", labeled=True)
    old, _ = read_xlsx(ROOT.parent / "EP1/deliveries/test1.xlsx", labeled=True)
    assert len(gold) == len(old) == 900
    assert [r.text for r in gold] == [r.text for r in old]
    assert sum(a.label == b.label for a, b in zip(gold, old)) == 432


def test_export_preserves_original_archive_and_has_exactly_897_labels(tmp_path):
    source = ROOT / "data/test2.xlsx"
    before = file_digest(source)
    records, _ = read_xlsx(source)
    predictions = [pred(r.excel_row, r.text, ("c1", "c234", "c5")[i % 3]) for i, r in enumerate(records)]
    target = tmp_path / "test2.xlsx"
    export_test2(source, target, predictions)
    checked, _ = read_xlsx(target, labeled=True)
    assert len(checked) == 897 and [r.text for r in checked] == [r.text for r in records]
    assert [r.label for r in checked] == [p["label"] for p in predictions]
    with ZipFile(source) as old, ZipFile(target) as new:
        assert old.namelist() == new.namelist()
        changed = [name for name in old.namelist() if old.read(name) != new.read(name)]
        assert len(changed) == 1 and changed[0].startswith("xl/worksheets/")
        assert b'xmlns:x14ac=' in new.read(changed[0])
    assert file_digest(source) == before
    with pytest.raises(ValueError, match="nova saída"):
        export_test2(source, target, predictions)
    with pytest.raises(ValueError):
        export_test2(source, tmp_path / "bad.xlsx", list(reversed(predictions)))
    with pytest.raises(ValueError, match="897"):
        export_test2(ROOT / "data/test1-rotulado-para-teste-estatistico.xlsx", tmp_path / "test1.xlsx", predictions)


def test_test1_comparison_copy_preserves_source_and_rejects_gold(tmp_path):
    source = ROOT / "data/test1.xlsx"
    before = file_digest(source)
    records, _ = read_xlsx(source)
    predictions = [pred(r.excel_row, r.text, "c5") for r in records]
    target = tmp_path / "test1-predito.xlsx"
    export_test1(source, target, predictions)
    checked, _ = read_xlsx(target, labeled=True)
    assert len(checked) == 900 and all(r.label == "c5" for r in checked)
    assert [r.text for r in checked] == [r.text for r in records]
    assert file_digest(source) == before
    with ZipFile(source) as old, ZipFile(target) as new:
        assert old.namelist() == new.namelist()
        assert sum(old.read(n) != new.read(n) for n in old.namelist()) == 1
    with pytest.raises(ValueError, match="gabarito"):
        export_test1(ROOT / "data/test1-rotulado-para-teste-estatistico.xlsx", tmp_path / "bad.xlsx", predictions)
