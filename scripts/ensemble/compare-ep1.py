"""Compare frozen soft-voting predictions with the actual EP1 delivery, offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ensemble_classifier.soft_pipeline import evaluate_soft
from gpt_classifier.data import digest, file_digest, read_jsonl, read_xlsx, write_json
from gpt_classifier.evaluation import align, compare, metrics

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT.parent / "EP1/deliveries/test1.xlsx")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/test1-rotulado-para-teste-estatistico.xlsx")
    parser.add_argument("--train", type=Path, default=ROOT / "data/train.xlsx")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use um novo diretório de comparação.")
    paths = {"baseline": args.baseline, "gold": args.gold, "train": args.train,
             "prediction_run": args.run / "run.json", "predictions": args.run / "predictions.jsonl"}
    hashes = {name: file_digest(path) for name, path in paths.items()}

    # Existing evaluator verifies the frozen artifact and all seven prediction files.
    individual = evaluate_soft(args.run, args.gold, args.output_dir / "individuals")
    gold, _ = read_xlsx(args.gold, labeled=True)
    baseline, _ = read_xlsx(args.baseline, labeled=True)
    train, _ = read_xlsx(args.train, labeled=True)
    predictions = read_jsonl(paths["predictions"])
    report = compare(gold, predictions, baseline, train)
    delivered = [{"excel_row": r.excel_row, "text_sha256": digest(r.text), "label": r.label}
                 for r in baseline]
    report["ep1"].update(metrics(gold, delivered))
    norberto = align(gold, read_jsonl(args.run / "members/norberto/predictions.jsonl"))
    report["ep1_delivery_matches_saved_norberto"] = all(
        r.label == label for r, label in zip(baseline, norberto))
    report["individuals"] = individual["metrics"]
    report["provenance"] = {name: {"path": str(path.resolve()), "sha256": hashes[name]}
                            for name, path in paths.items()}
    report["provenance"]["ensemble_artifact_sha256"] = individual["ensemble_artifact_sha256"]
    report["provenance"]["comparison_script_sha256"] = file_digest(Path(__file__))
    report["protocol"] = {
        "scope": "Comparação isolada das previsões existentes; sem ajuste de parâmetros.",
        "official_order_verified": False,
        "limitation": "O enunciado exige gerar teste2 (897) antes de avaliar teste1 (900). "
                      "Esta execução não comprova essa ordem; não constitui entrega oficial de teste2.",
        "historically_blind_test": False,
        "parameters_updated": False,
        "training_or_inference_performed": False,
    }
    if hashes != {name: file_digest(path) for name, path in paths.items()}:
        raise ValueError("Uma entrada foi alterada durante a comparação.")
    write_json(args.output_dir / "metrics.json", report, replace=False)
    print(json.dumps({"ep1_accuracy": report["ep1"]["accuracy"],
                      "ensemble_accuracy": report["ep2"]["accuracy"],
                      "ep1_delivery_matches_saved_norberto": report["ep1_delivery_matches_saved_norberto"],
                      **report["mcnemar"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
