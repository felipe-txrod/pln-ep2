from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .backend import BackendError, make_backend
from .data import Text, describe, file_digest, read_jsonl, read_xlsx, write_json, write_jsonl
from .delivery import export_test1, export_test2
from .evaluation import compare, error_cases
from .explain import explain
from .pipeline import environment, run
from .prompt import Config
from .study import develop
from .embeddings import DEFAULT_CACHE, prepare_model

ROOT = Path(__file__).resolve().parents[2]


def completed_run(directory: Path, *, rows: int):
    manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("rows") != rows:
        raise ValueError(f"O experimento deve estar concluído com {rows} linhas.")
    return manifest, read_jsonl(directory / "predictions.jsonl")


def parser():
    parser = argparse.ArgumentParser(description="EP2: classificador GPT de clareza e-SIC")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Inspeção local de planilhas; nenhuma chamada GPT")
    audit.add_argument("--output", type=Path)
    prepare = sub.add_parser("prepare-retrieval", help="Baixar encoder público fixado; não envia dados da base")
    prepare.add_argument("--retrieval-cache", type=Path, default=DEFAULT_CACHE)
    retrieval = sub.add_parser("retrieval-study", help="Diagnóstico local e piloto pareado de recuperação; treino somente")
    retrieval.add_argument("--train", type=Path, default=ROOT / "data/train.xlsx")
    retrieval.add_argument("--config", type=Path, default=ROOT / "configs/codex-hibrido-experimental.json")
    retrieval.add_argument("--output-dir", type=Path, required=True)
    retrieval.add_argument("--retrieval-cache", type=Path, default=DEFAULT_CACHE)
    retrieval.add_argument("--local-limit", type=int, default=300)
    retrieval.add_argument("--pilot-limit", type=int, default=12)
    retrieval.add_argument("--max-new-requests", type=int, default=8,
                           help="Teto compartilhado de processos Codex novos; padrão 8. Ampliação exige valor explícito.")
    retrieval.add_argument("--execute", action="store_true", help="Autorizar piloto GPT; padrão é diagnóstico local e prévia")
    retrieval.add_argument("--codex-bin", default="codex")
    retrieval.add_argument("--timeout", type=float, default=300)
    for name in ("develop", "predict"):
        p = sub.add_parser(name, help="Grade interna no treino" if name == "develop" else "Classificar sem enviar gabarito")
        p.add_argument("--train", type=Path, default=ROOT / "data/train.xlsx")
        p.add_argument("--config", type=Path, help="Padrão: configs/gpt.json (API) ou configs/codex.json (Codex)")
        p.add_argument("--backend", choices=("api", "codex"), help="Selecione a API ou o Codex CLI autenticado")
        p.add_argument("--model", help="Modelo explícito, gravado na configuração efetiva")
        p.add_argument("--batch-size", type=int, help="Textos por execução Codex (1 a 8); padrão da configuração")
        p.add_argument("--prompt-profile", choices=("original", "empirical"), help="Variante explícita do prompt")
        p.add_argument("--strategy", choices=("zero", "similar", "representative", "hybrid"), help="Substituir estratégia; zero ajusta per_class=0")
        p.add_argument("--retrieval-cache", type=Path, default=DEFAULT_CACHE)
        p.add_argument("--codex-bin", default="codex", help="Executável Codex CLI (backend codex)")
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--cache", type=Path, default=ROOT / ".cache/responses.sqlite3")
        p.add_argument("--dry-run", action="store_true", help="Preparar/auditar requests sem classificação remota ou previsões fictícias")
        p.add_argument("--max-new-requests", type=int, default=2000, help="Máximo de tentativas HTTP ou execuções Codex novas nesta invocação")
        p.add_argument("--retries", type=int, help="Exclusivo da API: retentativas transitórias (padrão 2)")
        p.add_argument("--timeout", type=float, help="Segundos por chamada: padrão 60 na API, 300 no Codex")
        p.add_argument("--min-interval", type=float, default=0.2)
        if name == "develop":
            p.add_argument("--limit", type=int, default=150, help="Número máximo de linhas da validação interna")
        else:
            p.add_argument("--input", type=Path, default=ROOT / "data/test2.xlsx")
            p.add_argument("--phase", choices=("test2", "statistical", "comparison"), default="test2",
                           help="comparison: teste1 separado, sem exigir teste2; statistical: comparação final vinculada a teste2")
            p.add_argument("--test2-run", type=Path, help="Obrigatório para statistical: execução teste2 concluída com a mesma configuração")
    evaluate = sub.add_parser("evaluate", help="Comparar EP1 x GPT em teste1 com McNemar; sem API")
    evaluate.add_argument("--gold", type=Path, default=ROOT / "data/test1-rotulado-para-teste-estatistico.xlsx")
    evaluate.add_argument("--baseline", type=Path, default=ROOT.parent / "EP1/deliveries/test1.xlsx")
    evaluate.add_argument("--train", type=Path, default=ROOT / "data/train.xlsx")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    delivery = sub.add_parser("export", help="Gerar XLSX de entrega teste2 ou cópia de teste1 para comparação")
    delivery.add_argument("--purpose", choices=("delivery", "comparison"), default="delivery",
                          help="delivery: teste2 oficial (897); comparison: cópia de teste1 (900)")
    delivery.add_argument("--input", type=Path, default=ROOT / "data/test2.xlsx")
    delivery.add_argument("--run", type=Path, required=True)
    delivery.add_argument("--output", type=Path, required=True)
    diagnostic = sub.add_parser("explain", help="Interpretação global/local de modelo lexical auxiliar")
    diagnostic.add_argument("--train", type=Path, default=ROOT / "data/train.xlsx")
    diagnostic.add_argument("--gold", type=Path, default=ROOT / "data/test1-rotulado-para-teste-estatistico.xlsx")
    diagnostic.add_argument("--run", type=Path, required=True)
    diagnostic.add_argument("--output", type=Path, required=True)
    return parser


def execute(args):
    if args.command == "prepare-retrieval":
        return prepare_model(args.retrieval_cache)
    if args.command == "retrieval-study":
        from .retrieval_study import retrieval_study
        train, info = read_xlsx(args.train, labeled=True)
        config = Config.load(args.config)
        if config.backend != "codex" or config.strategy != "hybrid":
            raise ValueError("retrieval-study exige configuração Codex com strategy=hybrid.")
        backend = make_backend(config, codex_bin=args.codex_bin, timeout=args.timeout,
                               max_requests=args.max_new_requests)
        return retrieval_study(train, config, output_dir=args.output_dir,
                               retrieval_cache=args.retrieval_cache, local_limit=args.local_limit,
                               pilot_limit=args.pilot_limit, execute=args.execute, backend=backend,
                               max_new_requests=args.max_new_requests, provenance={"train_info": info})
    if args.command == "audit":
        report = {p.name: describe(p, labeled=p.name not in {"test1.xlsx", "test2.xlsx"}) for p in sorted((ROOT / "data").glob("*.xlsx"))}
        if args.output:
            write_json(args.output, report, replace=False)
        return report
    if args.command in {"develop", "predict"}:
        config_path = args.config or ROOT / ("configs/codex.json" if args.backend == "codex" else "configs/gpt.json")
        config = Config.load(config_path)
        if args.backend and config.backend != args.backend:
            raise ValueError("--backend diverge do arquivo de configuração; use uma configuração correspondente.")
        if args.model:
            config = replace(config, model=args.model)
        if args.batch_size is not None:
            config = replace(config, batch_size=args.batch_size)
        if args.prompt_profile:
            config = replace(config, prompt_profile=args.prompt_profile)
        if args.strategy:
            config = replace(config, strategy=args.strategy,
                             per_class=0 if args.strategy == "zero" else max(1, config.per_class))
        train, train_info = read_xlsx(args.train, labeled=True)
        backend = make_backend(config, codex_bin=args.codex_bin, timeout=args.timeout, retries=args.retries,
                               max_requests=args.max_new_requests, min_interval=args.min_interval)
        provenance = {"train": str(args.train.resolve()), "train_info": train_info}
        if args.command == "develop":
            if args.limit < 3:
                raise ValueError("limit deve ser >=3.")
            return develop(train, config, output_dir=args.output_dir, cache_path=args.cache,
                           limit=args.limit, dry_run=args.dry_run, backend=backend, provenance=provenance,
                           retrieval_cache=args.retrieval_cache)
        if args.train.resolve() == args.input.resolve():
            raise ValueError("Treino e entrada de teste devem ser arquivos distintos.")
        texts, info = read_xlsx(args.input)  # Labels are never read into prediction records.
        expected = 897 if args.phase == "test2" else 900
        if len(texts) != expected:
            raise ValueError(f"A fase {args.phase} exige {expected} linhas.")
        if args.phase == "statistical":
            if not args.test2_run:
                raise ValueError("Primeiro conclua teste2; informe --test2-run na comparação estatística.")
            previous, _ = completed_run(args.test2_run, rows=897)
            if previous["config"] != config.to_dict() or previous["provenance"]["train_info"]["sha256"] != train_info["sha256"]:
                raise ValueError("A configuração e o treino devem ser os mesmos usados em teste2.")
            if previous["environment"] != environment():
                raise ValueError("Código e ambiente mudaram desde teste2. Preserve o método congelado para teste1.")
            if previous.get("backend") != backend.metadata():
                raise ValueError("Backend ou versão do Codex mudou desde teste2. Use o mesmo runtime congelado.")
            provenance["test2_run_fingerprint"] = previous["fingerprint"]
        provenance.update(input=str(args.input.resolve()), input_info=info, phase=args.phase)
        predictions = run(texts, train, config, output_dir=args.output_dir, cache_path=args.cache,
                          dry_run=args.dry_run, backend=backend, provenance=provenance,
                          retrieval_cache=args.retrieval_cache)
        return {"status": "dry_run" if args.dry_run else "completed", "rows": len(texts),
                "predictions": len(predictions), "output_dir": str(args.output_dir.resolve())}
    if args.command == "evaluate":
        manifest, predictions = completed_run(args.run, rows=900)
        phase = manifest.get("provenance", {}).get("phase")
        if phase not in {"statistical", "comparison"}:
            raise ValueError("A avaliação exige uma execução statistical ou comparison de teste1.")
        gold, _ = read_xlsx(args.gold, labeled=True)
        baseline, _ = read_xlsx(args.baseline, labeled=True)
        train, _ = read_xlsx(args.train, labeled=True)
        report = compare(gold, predictions, baseline, train)
        if phase == "comparison":
            report["caveat"] += " Execução isolada de comparação, sem vínculo a teste2; não usar o gabarito para escolher prompts, exemplos ou configurações."
        report["provenance"] = {"run_fingerprint": manifest["fingerprint"], "gold_sha256": file_digest(args.gold),
                                "baseline_sha256": file_digest(args.baseline), "train_sha256": file_digest(args.train)}
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "metrics.json", report)
        write_jsonl(args.output_dir / "errors.jsonl", error_cases(gold, predictions))
        p = report["mcnemar"]["p_exact_two_sided"]
        write_json(args.output_dir / "report-data.json", {"ep1_accuracy": report["ep1"]["accuracy"],
                   "ep2_accuracy": report["ep2"]["accuracy"], "mcnemar_chi2_corrected": report["mcnemar"]["chi2_continuity_corrected"],
                   "mcnemar_exact_p": p, "significantly_superior": report["mcnemar"]["significantly_superior"]})
        return report
    if args.command == "export":
        comparison = args.purpose == "comparison"
        rows = 900 if comparison else 897
        manifest, predictions = completed_run(args.run, rows=rows)
        phase = manifest.get("provenance", {}).get("phase")
        if comparison:
            if phase not in {"comparison", "statistical"}:
                raise ValueError("A cópia de comparação exige uma execução de teste1.")
            export_test1(args.input, args.output, predictions)
        else:
            if phase != "test2":
                raise ValueError("A entrega exige uma execução teste2.")
            export_test2(args.input, args.output, predictions)
        return {"output": str(args.output.resolve()), "sha256": file_digest(args.output), "rows": rows, "purpose": args.purpose}
    if args.command == "explain":
        _, predictions = completed_run(args.run, rows=900)
        train, _ = read_xlsx(args.train, labeled=True)
        gold, _ = read_xlsx(args.gold, labeled=True)
        explain(train, gold, predictions, args.output)
        return {"output": str(args.output.resolve()), "note": "Diagnóstico lexical auxiliar; não explica os mecanismos internos do GPT."}
    raise AssertionError("Comando desconhecido")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = execute(args)
    except KeyboardInterrupt:
        print("Interrompido. As respostas já salvas permanecem no cache; repita o comando para retomar.", file=sys.stderr)
        return 130
    except (ValueError, TypeError, OSError, KeyError, BackendError, ImportError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
