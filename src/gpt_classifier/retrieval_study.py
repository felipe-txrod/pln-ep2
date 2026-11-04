"""Train-only retrieval diagnostics and an explicitly bounded paired GPT study."""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from .backend import file_lock
from .data import Text, canonical, digest, group, write_json, write_jsonl
from .evaluation import metrics
from .pipeline import environment, run
from .selection import development_split, make_selector


def retrieval_diagnostic(bank, validation, config, *, retrieval_cache):
    selector = make_selector(bank, config, retrieval_cache=retrieval_cache)
    selector.prepare_targets([r.text for r in validation])
    results, details, selections = {}, [], {}
    for strategy in ("similar", "hybrid"):
        predictions, purity = [], []
        selections[strategy] = []
        for target in validation:
            scores = selector.relevance(target.text, strategy=strategy, balanced=False).copy()
            target_group = group(target.text)
            scores[np.array([g == target_group for g in selector.group_ids])] = -np.inf
            top = np.argsort(-scores, kind="stable")[:5]
            predictions.append({"excel_row": target.excel_row, "text_sha256": digest(target.text),
                                "label": selector.records[top[0]].label})
            purity.append(sum(selector.records[i].label == target.label for i in top) / len(top))
            selected = selector.select(target.text, strategy=strategy, per_class=config.per_class)
            selections[strategy].append([e.excel_row for e in selected])
            details.append({"strategy": strategy, "excel_row": target.excel_row,
                            "nearest_excel_rows": [selector.records[i].excel_row for i in top],
                            "example_excel_rows": selections[strategy][-1], "top5_label_purity": purity[-1]})
        results[strategy] = {"nearest_neighbor_proxy": metrics(validation, predictions),
                             "mean_top5_label_purity": float(np.mean(purity))}
    changed = sum(a != b for a, b in zip(selections["similar"], selections["hybrid"]))
    return {"rows": len(validation), "results": results, "selection": selector.summary,
            "targets_with_changed_demonstrations": changed, "embedding_cache": selector.embeddings.stats,
            "caveat": "Vizinhos globais são diagnóstico de recuperação, não previsões GPT. A seleção do prompt permanece balanceada por classe. Não selecionar configuração por esse proxy."}, details


def retrieval_study(records, config, *, output_dir: Path, retrieval_cache: Path,
                    local_limit: int = 300, pilot_limit: int = 12, execute: bool = False,
                    backend, max_new_requests: int = 8, provenance=None):
    if config.strategy != "hybrid" or config.backend != "codex":
        raise ValueError("Estudo exige configuração Codex/hybrid.")
    if not 3 <= pilot_limit <= local_limit:
        raise ValueError("Exija 3 <= pilot_limit <= local_limit.")
    if type(max_new_requests) is not int or max_new_requests < 0:
        raise ValueError("O orçamento de chamadas deve ser inteiro não negativo.")
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {"config": config.to_dict(), "local_limit": local_limit, "pilot_limit": pilot_limit,
                "max_new_requests": max_new_requests, "environment": environment(),
                "data_sha256": digest(canonical([(r.excel_row, r.text, r.label) for r in records])),
                "provenance": provenance or {}}
    with file_lock(output_dir / ".retrieval-study.lock"):
        identity_path = output_dir / "study-plan.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Diretório pertence a outro estudo/código/amostra; escolha outro destino.")
        write_json(identity_path, identity)
        bank, validation = development_split(records, seed=config.seed, limit=local_limit)
        pilot_bank, pilot = development_split(records, seed=config.seed, limit=pilot_limit)
        assert [r.excel_row for r in bank] == [r.excel_row for r in pilot_bank]
        write_json(output_dir / "split.json", {"bank_excel_rows": [r.excel_row for r in bank],
                   "local_validation_excel_rows": [r.excel_row for r in validation],
                   "pilot_excel_rows": [r.excel_row for r in pilot], "pilot_class_counts": dict(Counter(r.label for r in pilot)),
                   "method": "Primeiro fold agrupado estratificado (5, seed fixa); banco exclui todo o fold, inclusive registros não amostrados."})
        diagnostic_path = output_dir / "retrieval-diagnostic.json"
        if not diagnostic_path.exists():
            diagnostic, details = retrieval_diagnostic(bank, validation, config, retrieval_cache=retrieval_cache)
            write_json(diagnostic_path, diagnostic)
            write_jsonl(output_dir / "retrieval-details.jsonl", details)
        else:
            diagnostic = json.loads(diagnostic_path.read_text())
        # Gold lives only in local evaluation; run receives Text, never Example, as targets.
        write_jsonl(output_dir / "local-pilot-gold.jsonl", pilot)
        arms = [("lexical", replace(config, strategy="similar")), ("hybrid", config)]
        previews = []
        for name, candidate in arms:
            directory = output_dir / (name + "-preview")
            run([Text(r.excel_row, r.text) for r in pilot], bank, candidate, output_dir=directory,
                cache_path=output_dir / "responses.sqlite3", dry_run=True, backend=backend,
                retrieval_cache=retrieval_cache, provenance=provenance)
            manifest = json.loads((directory / "run.json").read_text())
            previews.append({"arm": name, "unique_requests": manifest["unique_requests"],
                             "request_bytes": manifest["total_request_bytes"]})
        planned = sum(p["unique_requests"] for p in previews)
        report_path = output_dir / "report.json"
        if report_path.exists() and not execute:
            previous = json.loads(report_path.read_text())
            if previous.get("status") == "completed":
                return previous  # A later preview must not erase measured results.
        report = {"status": "preview", "diagnostic": diagnostic, "previews": previews,
                  "pilot_rows": len(pilot), "planned_requests": planned, "max_new_requests": max_new_requests,
                  "caveat": "Comparação exploratória na validação interna do treino; não é avaliação independente em teste externo nem seleciona automaticamente método de produção. Ambos os braços usam o mesmo prompt e lote; difere somente a recuperação. Nenhum teste1/teste2 foi usado."}
        write_json(report_path, report)
        if not execute:
            return report
        if planned > max_new_requests:
            raise ValueError(f"Prévia exige {planned} chamadas, acima do teto {max_new_requests}; nenhuma classificação iniciada.")
        scores = []
        initial_attempts = backend.attempts
        initial_usage = backend.usage.copy()
        try:
            for name, candidate in arms:
                started = time.monotonic()
                directory = output_dir / name
                predictions = run([Text(r.excel_row, r.text) for r in pilot], bank, candidate,
                                  output_dir=directory, cache_path=output_dir / "responses.sqlite3",
                                  backend=backend, retrieval_cache=retrieval_cache, provenance=provenance)
                manifest = json.loads((directory / "run.json").read_text())
                scores.append({"arm": name, "config": candidate.to_dict(), "metrics": metrics(pilot, predictions),
                               "new_usage": manifest["new_usage"], "new_requests": manifest["new_backend_attempts"],
                               "elapsed_seconds": time.monotonic() - started})
                write_json(output_dir / "scores.json", scores)
            report["status"] = "completed"
        except BaseException:
            report["status"] = "incomplete"
            raise
        finally:
            report.update(scores=scores, new_codex_processes=backend.attempts - initial_attempts,
                          new_usage={k: v - initial_usage[k] for k, v in backend.usage.items()})
            write_json(output_dir / "report.json", report)
        return report
