"""Small paired train-only pilot; at most ten new Codex processes.

Run from EP2 with: .venv/bin/python scripts/experiments/gpt/pilot-efficiency.py --execute
The default is a dry run. This is a smoke/triage experiment, not model selection.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from gpt_classifier.backend import make_backend
from gpt_classifier.data import Text, file_digest, read_xlsx, write_json, write_jsonl
from gpt_classifier.evaluation import metrics
from gpt_classifier.pipeline import run
from gpt_classifier.prompt import Config
from gpt_classifier.selection import development_split

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/pilot-efficiency")
    args = parser.parse_args()
    train_path = ROOT / "data/train_ep1.xlsx"
    records, _ = read_xlsx(train_path, labeled=True)
    base = Config.load(ROOT / "configs/codex.json")
    bank, validation = development_split(records, seed=base.seed, limit=6)
    if dict(Counter(r.label for r in validation)) != {"c1":2,"c234":2,"c5":2}:
        raise ValueError("O piloto exige exatamente dois exemplos por classe.")
    backend = make_backend(base, max_requests=10)
    provenance = {"purpose":"train-only paired pilot, not model selection", "train_sha256":file_digest(train_path)}
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "split.json", {"bank_excel_rows":[r.excel_row for r in bank],
               "validation_excel_rows":[r.excel_row for r in validation], "seed":base.seed,
               "caveat":"All labels come from training only. Entire validation fold is excluded from the example bank."})
    write_jsonl(directory / "local-validation-gold.jsonl", validation)
    arms = [("original-single",replace(base,batch_size=1,prompt_profile="original")),
            ("original-batch3",replace(base,batch_size=3,prompt_profile="original")),
            ("empirical-batch3",replace(base,batch_size=3,prompt_profile="empirical"))]
    results, predicted_by_arm = [], {}
    try:
        for name, config in arms:
            start = time.monotonic()
            arm_dir = directory / name
            predictions = run([Text(r.excel_row,r.text) for r in validation],bank,config,
                              output_dir=arm_dir,cache_path=directory / "cache.sqlite3",
                              dry_run=not args.execute,backend=backend,provenance=provenance)
            manifest = json.loads((arm_dir / "run.json").read_text())
            result = {"arm":name,"config":config.to_dict(),"elapsed_seconds":time.monotonic()-start,
                      "unique_requests":manifest["unique_requests"],"status":manifest["status"]}
            if args.execute:
                result.update(metrics=metrics(validation,predictions),new_usage=manifest["new_usage"],
                              attempts=manifest["new_backend_attempts"])
                predicted_by_arm[name] = [p["label"] for p in predictions]
            results.append(result)
            write_json(directory / "scores.json", results)
            print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        report = {"rows":len(validation),"class_counts":dict(Counter(r.label for r in validation)),
                  "new_codex_processes":backend.attempts,"total_usage":backend.usage,
                  "results":results,"note":"Six paired targets are insufficient to select a production model or establish accuracy gains."}
        if "original-single" in predicted_by_arm:
            baseline = predicted_by_arm["original-single"]
            report["agreements_with_single"] = {name:sum(a==b for a,b in zip(baseline,values))/len(baseline)
                                                for name,values in predicted_by_arm.items()}
        write_json(directory / "report.json",report)


if __name__ == "__main__":
    main()
