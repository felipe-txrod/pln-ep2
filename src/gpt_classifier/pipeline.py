from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .backend import API_BASE, Cache, make_backend, file_lock
from .data import Example, Text, canonical, digest, file_digest, write_json, write_jsonl
from .prompt import Config, build_request, parse_response
from .selection import Selector


def environment():
    packages = {}
    for name in ("openai", "openpyxl", "numpy", "scipy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {"python": platform.python_version(), "packages": packages,
            "source_sha256": {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}


def run(records: list[Text], bank: list[Example], config: Config, *, output_dir: Path,
        cache_path: Path, dry_run: bool = False, backend=None, provenance: dict | None = None):
    if not records or any(type(r) is not Text for r in records):
        raise ValueError("Inferência exige registros sem rótulos.")
    if len({r.excel_row for r in records}) != len(records):
        raise ValueError("IDs de linha duplicados.")
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = backend or make_backend(config)
    if getattr(backend, "kind", config.backend) != config.backend:
        raise ValueError("O backend não corresponde à configuração.")
    backend_info = backend.metadata() if hasattr(backend, "metadata") else {"kind": config.backend, "endpoint": API_BASE}
    env = environment()
    identity = {"config": config.to_dict(), "endpoint": backend_info["endpoint"], "backend": backend_info, "dry_run": dry_run,
                "texts_sha256": digest(canonical([(r.excel_row, r.text) for r in records])),
                "bank_sha256": digest(canonical([(r.excel_row, r.text, r.label) for r in bank])),
                "environment": env, "provenance": provenance or {}}
    fingerprint = digest(canonical(identity))
    with file_lock(output_dir / ".lock"), file_lock(cache_path.with_suffix(".lock")):
        manifest_path = output_dir / "run.json"
        history = []
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("fingerprint") != fingerprint:
                raise ValueError("Diretório pertence a outro experimento/configuração/código; escolha outro destino.")
            history = old.get("attempt_history", [])
        manifest = {**identity, "fingerprint": fingerprint, "status": "preparing",
                    "started_at": datetime.now(timezone.utc).isoformat(), "rows": len(records)}
        write_json(manifest_path, manifest)
        selector = None if config.strategy == "zero" else Selector(bank, max_chars=config.max_example_chars)
        requests, plans = [], []
        for record in records:
            examples = [] if selector is None else selector.select(record.text, strategy=config.strategy, per_class=config.per_class)
            request, truncated = build_request(record, examples, config)
            if config.backend == "codex":
                request["runtime"] = backend_info
            if len(canonical(request).encode("utf-8")) > config.max_request_bytes:
                raise ValueError("Request com metadados do backend excede max_request_bytes.")
            requests.append(request)
            plans.append({"excel_row": record.excel_row, "text_sha256": record.text_sha256,
                          "request_key": Cache.key(request), "example_excel_rows": [e.excel_row for e in examples],
                          "text_truncated": truncated, "request_bytes": len(canonical(request).encode("utf-8"))})
        write_jsonl(output_dir / "plan.jsonl", plans)
        write_json(output_dir / "request-preview.json", requests[0])
        write_json(output_dir / "config.json", config.to_dict())
        manifest.update(selection=None if selector is None else selector.summary,
                        unique_requests=len({p["request_key"] for p in plans}),
                        truncated_rows=sum(p["text_truncated"] for p in plans),
                        total_request_bytes=sum(p["request_bytes"] for p in plans))
        if dry_run:
            manifest["status"] = "dry_run"
            write_json(manifest_path, manifest)
            return []
        initial_attempts = backend.attempts
        initial_usage = backend.usage.copy()
        cache = Cache(cache_path)
        predictions, hits = [], 0
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        try:
            for index, (request, plan) in enumerate(zip(requests, plans), 1):
                response = cache.get(request)
                cached = response is not None
                if cached:
                    hits += 1
                else:
                    response = backend.create(request)
                    # Persist received failures for diagnosis/usage but never as predictions.
                    try:
                        parse_response(response)
                    except ValueError:
                        write_json(output_dir / "failed-response.json", {"plan": plan, "response": response})
                        raise
                    cache.put(request, response)
                predictions.append({**plan, "label": parse_response(response), "cache_hit": cached,
                                    "response_id": response.get("id"), "response_model": response.get("model"),
                                    "backend": config.backend,
                                    "model_is_requested_alias": response.get("codex", {}).get("model_is_requested_alias", False),
                                    "usage": response.get("usage")})
                if index == 1 or index % 25 == 0 or index == len(records):
                    print(f"{index}/{len(records)} classificados; {hits} cache; {backend.attempts - initial_attempts} tentativas novas ({config.backend})", file=sys.stderr)
            write_jsonl(output_dir / "predictions.jsonl", predictions)
            manifest["status"] = "completed"
            return predictions
        except BaseException as exc:
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                            error_type=type(exc).__name__, completed_rows=len(predictions))
            raise
        finally:
            cache.close()
            new_usage = {key: value - initial_usage[key] for key, value in backend.usage.items()}
            finished = datetime.now(timezone.utc).isoformat()
            attempts = backend.attempts - initial_attempts
            history.append({"finished_at": finished, "status": manifest["status"],
                            "api_attempts": attempts if config.backend == "api" else 0,
                            "backend_attempts": attempts, "usage": new_usage})
            manifest.update(cache_hits=hits, new_api_attempts=attempts if config.backend == "api" else 0,
                            new_backend_attempts=attempts,
                            new_usage=new_usage, attempt_history=history, finished_at=finished)
            write_json(manifest_path, manifest)
