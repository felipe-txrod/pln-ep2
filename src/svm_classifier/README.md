<a id="frozen-ep1-svm-classifier"></a>

# Classificador SVM congelado do EP1

Inferência em CPU com `LinearSVC` treinado, sem comandos de treinamento ou
ajuste automático. Carrega só por `load()` ou `predict()`. Importar o pacote
não importa scikit-learn, NumPy, PyTorch ou bibliotecas GPU.

<a id="preserved-configuration"></a>

## Configuração preservada

- `LinearSVC`: `C=0.05`, `dual="auto"`, `max_iter=10000`, `tol=1e-4`, semente 42.
- TF-IDF de palavras: ngramas 1–2, `min_df=2`, `max_df=0.98`, até 60,000 atributos.
- TF-IDF de caracteres: `char_wb`, ngramas 3–5, `min_df=3`, até 50,000 atributos.
- Frequência sublinear, normalização L2 e matrizes FP32.
- Pré-processamento: decodificação HTML, espaços normalizados, minúsculas;
  acentos, pontuação e negação preservados.
- Classes: `c1`, `c234`, `c5`; duas threads CPU por padrão.

O artefato original é `EP1/artifacts/svm-fixed-holdout-20260923.joblib`,
treinado em 16,093 respostas. A cópia independente versionada é
[`models/svm/model.joblib`](../../models/svm/README.md), com `config.json`,
`metadata.json` e manifestos. O [manifesto congelado das fontes](frozen_manifest.json)
fixa hashes e parâmetros. O caminho anterior `artifacts/svm-ep1/` permanece
alternativa quando não há pacote publicado.

<a id="environment-and-usage"></a>

## Ambiente e uso

`requirements-lexical.txt` fixa dependências EP1. Use ambiente Python 3.12
separado: o carregador exige versões salvas de NumPy, SciPy, scikit-learn e
joblib. O arquivo geral de dependências EP2 usa outro joblib e não substitui
esses requisitos. Bibliotecas GPU e pandas são desnecessários.

Na raiz do EP2 com essas dependências instaladas:

```bash
PYTHONPATH=src python -m svm_classifier predict \
  --input data/test2.xlsx --output-dir results/svm-development-test2
```

```python
from svm_classifier import SVMClassifier

classifier = SVMClassifier()
predictions = classifier.predict(["Example public response."])
label = predictions[0].label
scores = predictions[0].decision_scores
```

A CLI grava `predictions.jsonl` e `run.json`, preservando IDs/hashes sem
alterar planilha ou ler gabarito. `info` inspeciona manifesto/presença de
arquivos sem carregar modelo. `--artifact` e `SVM_ARTIFACT_PATH` selecionam
outra cópia com mesmo hash. `SVM_BUNDLE_DIR` seleciona o diretório de metadados,
configuração e `model.joblib` padrão.

SVM fornece rótulos e margens, não probabilidades. Não aplique softmax para
simular probabilidades calibradas. No ensemble, usa rótulos, consenso ou
maioria estrita e árbitro configurado.

<a id="serialization-and-validation"></a>

## Serialização e validação

O carregador confere o hash antes da desserialização e remapeia só a referência
histórica `clarity.data.normalize_text` para a função local equivalente de
pré-processamento. A cópia não exige imports EP1 nem seu diretório. A adaptação
usa o [carregador de arrays joblib 1.5.2](https://github.com/joblib/joblib/blob/1.5.2/joblib/numpy_pickle.py).
Vocabulários, IDF e coeficientes vêm do artefato treinado.

O pacote já concluiu inferência em CPU nas 900 respostas de teste1, com
**49.44% de acurácia (445/900)**. IDs, textos, parâmetros e hashes foram
verificados; veja a [avaliação completa](../../docs/results/lexical-test1.md).
