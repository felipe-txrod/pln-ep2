# EP1 and EP2 dataset overlap

[Results index](README.md) · [Machine-readable audit](data/dataset-overlap.json)

## Assignment and dataset identity

The [EP2 assignment](../reference/assignment.md) explicitly reuses the EP1
training corpus. It introduces test2 (897 rows) as the official prediction
deliverable and provides test1 labels (900 rows) only for the final paired
statistical comparison with the EP1 submission.

The following training files are byte-identical, with the same SHA-256,
20,092 rows, response texts, labels, and row order:

- `EP1/data/train.xlsx`
- `EP2/data/train_ep1.xlsx`
- `EP2/data/train_ep2.xlsx`

The two unlabeled test1 workbooks are also byte-identical. The labeled test1
workbook contains exactly the same 900 response texts in the same order.
Training copies are alternative filenames for one corpus; concatenating them
would repeat the same examples.

## Text overlap across partitions

Exact matching compares complete response strings, without changing them.
Normalized matching uses the EP1 grouping rule: HTML entity decoding, whitespace
collapse, trimming, and lowercase. It preserves accents and punctuation.

| Reference corpus | Compared partition | Exact matching rows | Normalized matching rows |
|---|---|---:|---:|
| Training, 20,092 rows | Test1, 900 rows | 88 (9.78%) | 105 (11.67%) |
| Training, 20,092 rows | Test2, 897 rows | 93 (10.37%) | 107 (11.93%) |
| Test1, 900 rows | Test2, 897 rows | 37 (4.12%) | 47 (5.24%) |

Counts refer to rows in the compared partition, each counted once regardless
of how many matches exist in the reference corpus. Exact matches are included
in normalized counts. Training/test1 share 72 exact or 83 normalized unique
texts; training/test2 share 67 exact or 73 normalized unique texts.

## Repeated texts and label disagreement

Training contains 18,432 unique exact texts or 18,143 normalized texts.
There are 291 exact-text groups with different labels (1,682 rows), increasing
to 316 normalized groups (1,975 rows).

Among the 105 test1 rows with normalized matches in training:

- 21 have a gold label matching every training occurrence of that text.
- 72 have both matching and different labels among training occurrences.
- 12 have no training occurrence with their gold label.

The labeled test1 itself contains 10 exact-text groups with different labels
(24 rows), or 11 normalized groups (31 rows). Matching a text therefore does
not guarantee recovering its label. The workbooks expose response text and
clarity only; they cannot establish whether repeated responses correspond to
the same request, author, or annotation context.

## Interpretation and reproduction

The supplied files have textual overlap between training and both test sets;
response texts are not fully disjoint. These counts describe the provided data,
independently of any classifier. Official scores include all prescribed rows;
the overlap and annotation ambiguity should be disclosed when interpreting
generalization and paired significance tests. Test1 gold labels remain reserved
for comparison and must not be added to training, retrieval, or tie-breaking.

The CPU audit performs no training or model inference, does not modify source
workbooks, and publishes only aggregates and workbook hashes. Run it with
openpyxl available, from the EP2 root, selecting a new output file:

```bash
PYTHONPATH=src python3 scripts/diagnostics/audit-dataset-overlap.py \
  --ep1-root ../EP1 \
  --output results/dataset-overlap-new/summary.json
```

Seven workbooks and all 21 pairs were inspected. Matching is exact or normalized;
near-duplicates and semantic similarity were not evaluated. Synthetic tests
verify occurrence counts, normalization, conflicting labels, and ordered identity.
