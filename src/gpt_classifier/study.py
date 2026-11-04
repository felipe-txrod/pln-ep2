from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .backend import file_lock
from .data import Text, canonical, digest, write_json, write_jsonl
from .evaluation import metrics
from .pipeline import run
from .prompt import Config
from .selection import development_split


def develop(records, config: Config, *, output_dir: Path, cache_path: Path,
            limit: int = 150, dry_run: bool = False, backend=None, provenance=None, retrieval_cache=None):
    bank, validation = development_split(records, seed=config.seed, limit=limit)
    configs = [replace(config, strategy="zero", per_class=0),
               replace(config, strategy="representative", per_class=1),
               replace(config, strategy="similar", per_class=1),
               replace(config, strategy="similar", per_class=2)]
    if config.strategy == "hybrid":
        configs.append(replace(config, strategy="hybrid", per_class=1))
    identity = {"configs": [c.to_dict() for c in configs], "limit": limit, "dry_run": dry_run,
                "data_sha256": digest(canonical([(r.excel_row, r.text, r.label) for r in records]))}
    output_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(output_dir / ".study.lock"):
        plan = output_dir / "study-plan.json"
        if plan.exists() and json.loads(plan.read_text(encoding="utf-8")) != identity:
            raise ValueError("Este diretório já contém outra grade/amostra. Use um novo destino.")
        write_json(plan, identity)
        write_json(output_dir / "split.json", {"train_excel_rows": [r.excel_row for r in bank],
                   "validation_excel_rows": [r.excel_row for r in validation],
                   "method": "Primeiro fold de StratifiedGroupKFold(5), seguido de subamostra estratificada da validação; treino exclui todo o fold."})
        results = []
        for index, candidate in enumerate(configs):
            directory = output_dir / f"candidate-{index:02d}"
            predictions = run([Text(r.excel_row, r.text) for r in validation], bank, candidate,
                              output_dir=directory, cache_path=cache_path, dry_run=dry_run,
                              backend=backend, provenance=provenance, retrieval_cache=retrieval_cache)
            if not dry_run:
                manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
                result = {"candidate": directory.name, "config": candidate.to_dict(),
                          "metrics": metrics(validation, predictions),
                          "new_usage": manifest["new_usage"], "unique_requests": manifest["unique_requests"]}
                results.append(result)
                write_json(output_dir / "scores.json", results)
        if dry_run:
            return {"status": "dry_run", "validation_rows": len(validation), "candidates": len(configs)}
        best = max(results, key=lambda r: r["metrics"]["accuracy"])
        write_json(output_dir / "selected-config.json", best["config"])
        write_json(output_dir / "selection.json", {"criterion": "accuracy; empate favorece primeira configuração da grade",
                   "winner": best["candidate"], "development_result": best["metrics"],
                   "caveat": "Resultado de seleção exploratória na mesma amostra; não é estimativa independente nem teste1."})
        return best
