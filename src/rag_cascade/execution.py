"""Prebuilt-request execution using the existing cache, Codex transport and batch contract."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time

from cascade_classifier.gpt import estimated_cost, token_usage
from gpt_classifier.backend import BackendError, Cache, file_lock
from gpt_classifier.batching import prepare_batches
from gpt_classifier.codex_backend import CodexBackend
from gpt_classifier.data import LABELS, canonical, write_json
from gpt_classifier.prompt import validate_response


class RequestRunner:
    """A preflight miss count and explicit execute flag protect every classification call."""

    def __init__(self, config, cache: Path, *, backend_factory=None):
        self.config, self.cache_path = config, cache
        self.backend_factory = backend_factory or (lambda: CodexBackend(
            executable=os.environ.get("CASCADE_CODEX_BIN", "codex"), timeout=config.timeout,
            min_interval=0, max_requests=1))

    def run(self, items: list[dict], output: Path, *, execute=False, max_new_requests=0):
        if type(max_new_requests) is not int or max_new_requests < 0:
            raise ValueError("max_new_requests must be an explicit nonnegative integer.")
        if len({item["excel_row"] for item in items}) != len(items):
            raise ValueError("Duplicate target IDs in the request plan.")
        with file_lock(self.cache_path.with_suffix(".lock")):
            cache = Cache(self.cache_path)
            try:
                return self._run(items, output, cache, execute, max_new_requests)
            finally:
                cache.close()

    def _run(self, items, output, cache, execute, limit):
        db, config = cache.connection, self.config
        db.execute("CREATE TABLE IF NOT EXISTS cascade_decisions (key TEXT PRIMARY KEY, request TEXT NOT NULL, decision TEXT NOT NULL)")
        db.commit()
        decisions, missing, indexes = {}, [], []
        for index, item in enumerate(items):
            request = item["request"]
            key = Cache.key(request)
            if item["request_key"] != key:
                raise ValueError("Request changed after planning.")
            saved = db.execute("SELECT request,decision FROM cascade_decisions WHERE key=?", (key,)).fetchone()
            if saved:
                value = json.loads(saved[1])
                if saved[0] != canonical(request) or value["label"] not in LABELS:
                    raise ValueError("Invalid item cache provenance.")
                decisions[index] = {**value, "cache_hit": True}
            else:
                missing.append(request)
                indexes.append(index)
        batches = prepare_batches(missing, batch_size=config.batch_size, max_bytes=config.max_request_bytes)
        raw_cache = {Cache.key(request): cache.get(request) for request, _ in batches}
        needed = sum(raw_cache[Cache.key(request)] is None for request, _ in batches)
        preview = {"status": "preview", "targets": len(items), "item_cache_hits": len(decisions),
                   "new_batches": needed, "max_new_requests": limit, "classification_executed": False}
        if not execute:
            return {}, preview
        if needed > limit:
            raise ValueError(f"Plan requires {needed} new batches, exceeding the explicit limit {limit}; no request sent.")
        if items and any(item["request"]["runtime"] != self.backend_factory().metadata() for item in items[:1]):
            raise ValueError("Codex runtime changed after planning.")
        output.mkdir(parents=True, exist_ok=True)
        identity = {"keys": [item["request_key"] for item in items], "config": config.to_dict()}
        if (output / "plan.json").exists() and json.loads((output / "plan.json").read_text()) != identity:
            raise ValueError("Output belongs to another immutable request plan.")
        write_json(output / "plan.json", identity)
        lock, circuit = threading.Lock(), threading.Event()
        new_attempts, started = 0, time.monotonic()

        def call(job):
            nonlocal new_attempts
            request, mapping = job
            key = Cache.key(request)
            path = output / "attempts" / f"{key}.json"
            history = json.loads(path.read_text()) if path.exists() else []
            response = raw_cache[key]
            if response is not None:
                return request, mapping, response, history
            response_path = output / "responses" / f"{key}.json"
            if history and history[-1]["status"] == "completed":
                response = json.loads(response_path.read_text())
                validate_response(request, response)
                return request, mapping, response, history
            if history and history[-1]["status"] == "started":
                history[-1]["status"] = "interrupted_unknown_usage"
                write_json(path, history)
            for attempt in range(len(history), config.retries + 1):
                if circuit.is_set():
                    break
                with lock:
                    if new_attempts >= limit:
                        break
                    new_attempts += 1
                entry = {"attempt": attempt + 1, "status": "started", "usage_known": False}
                history.append(entry)
                write_json(path, history)
                begin = time.monotonic()
                backend = self.backend_factory()
                try:
                    response = backend.create(request)
                    validate_response(request, response)
                    entry.update(status="completed", usage=token_usage(response), usage_known=True)
                    write_json(response_path, response)
                except (BackendError, ValueError, OSError) as error:
                    response = None
                    entry.update(status="failed", error_type=type(error).__name__, usage=backend.usage.copy(),
                                 usage_known=bool(sum(backend.usage.values())))
                finally:
                    entry["latency_seconds"] = time.monotonic() - begin
                    write_json(path, history)
                if response is not None:
                    break
                if attempt < config.retries:
                    time.sleep(min(2 ** attempt, 4))
            if response is None:
                circuit.set()
            return request, mapping, response, history

        with ThreadPoolExecutor(max_workers=config.workers) as workers:
            for request, mapping, response, history in workers.map(call, batches):
                key = Cache.key(request)
                if response is None:
                    for _, positions in mapping:
                        for position in positions:
                            decisions[indexes[position]] = {"label": None, "status": "fallback", "request_key": key}
                    continue
                parsed = validate_response(request, response)
                cache.put(request, response)
                usage = token_usage(response)
                latency = sum(row.get("latency_seconds", 0) for row in history)
                for identifier, positions in mapping:
                    for position in positions:
                        index = indexes[position]
                        item_key = Cache.key(items[index]["request"])
                        receipt = {"status": "completed", "label": parsed[identifier], "item_request_key": item_key,
                                   "request_key": key, "usage_allocation": {name: value / len(mapping) for name, value in usage.items()},
                                   "latency_allocation_seconds": latency / len(mapping)}
                        db.execute("INSERT OR IGNORE INTO cascade_decisions VALUES (?,?,?)",
                                   (item_key, canonical(items[index]["request"]), canonical(receipt)))
                        decisions[index] = {**receipt, "cache_hit": raw_cache[key] is not None}
                db.commit()
        histories = [row for path in (output / "attempts").glob("*.json") for row in json.loads(path.read_text())]
        usage = {key: sum(row.get("usage", {}).get(key, 0) for row in histories)
                 for key in ("input_tokens", "output_tokens", "cached_input_tokens")}
        report = {**preview, "status": "completed", "classification_executed": True, "new_attempts": new_attempts,
                  "total_output_attempts": len(histories), "unknown_usage_attempts": sum(not row.get("usage_known", False) for row in histories),
                  "failed_attempts": sum(row["status"] != "completed" for row in histories), "usage": usage,
                  "estimated_api_equivalent_usd": estimated_cost(usage, config), "wall_seconds": time.monotonic() - started,
                  "summed_request_latency_seconds": sum(row.get("latency_seconds", 0) for row in histories),
                  "billing_note": "Standard API-equivalent estimate; unknown failed usage is excluded; not a subscription invoice."}
        result = {item["excel_row"]: decisions[index] for index, item in enumerate(items)}
        write_json(output / "decisions.json", {str(key): value for key, value in result.items()})
        write_json(output / "run.json", report)
        return result, report
