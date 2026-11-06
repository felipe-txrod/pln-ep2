"""Local-only commands: extract frozen features, audit retrieval and replay logged decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from gpt_classifier.data import Text, canonical, digest, file_digest, write_json
from gpt_classifier.selection import Selector
from .config import ResearchConfig
from .data import load_data


def eligible_selector(data):
    return Selector([row for row in data.bank if len(row.text) <= data.old_config.max_example_chars],
                    max_chars=data.old_config.max_example_chars)


def extract(config, output):
    import numpy as np
    from .norberto_embeddings import FineTunedEmbeddings
    data = load_data(config)
    selector = eligible_selector(data)
    encoder = FineTunedEmbeddings(config)
    bank = [Text(row.excel_row, row.text) for row in selector.records]
    encoded = encoder.encode([*bank, *data.targets])
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "norberto-vectors.npz"
    if archive.exists():
        raise FileExistsError("Use a fresh feature-export directory; cached embeddings are reusable.")
    np.savez_compressed(archive, bank_ids=np.array([row.excel_row for row in bank]),
                        target_ids=np.array([row.excel_row for row in data.targets]), **encoded)
    metadata = {"status": "complete", "encoder": encoder.metadata, "namespace": encoder.namespace,
                "views": encoder.views, "stats": encoder.stats, "bank_rows": len(bank), "target_rows": len(data.targets),
                "bank_sha256": digest(canonical([(row.excel_row, row.text_sha256) for row in bank])),
                "target_sha256": digest(canonical([(row.excel_row, row.text_sha256) for row in data.targets])),
                "source": data.identities, "archive_sha256": file_digest(archive),
                "classifier_updates": 0, "remote_calls": 0}
    write_json(output / "embeddings.json", metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline RAG diagnostics; no GPT backend is exposed")
    parser.add_argument("--config", type=Path, default=Path("configs/rag-research.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    encode = commands.add_parser("encode")
    encode.add_argument("--output-dir", type=Path, required=True)
    retrieval = commands.add_parser("retrieval")
    retrieval.add_argument("--embeddings", type=Path, required=True)
    retrieval.add_argument("--output-dir", type=Path, required=True)
    interventions = commands.add_parser("interventions")
    interventions.add_argument("--output-dir", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--retrieval", type=Path, required=True)
    report.add_argument("--interventions", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = ResearchConfig.load(args.config)
        if args.command == "encode":
            result = extract(config, args.output_dir)
        elif args.command == "retrieval":
            from .retrieval_study import run
            result = run(config, args.embeddings, args.output_dir)
        elif args.command == "report":
            from .reporting import run
            result = run(config, args.retrieval, args.interventions, args.output_dir)
        else:
            from .acceptance import run
            result = run(config, args.output_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
