<a id="frozen-svm"></a>

# SVM congelado

`model.joblib` contém pipeline treinado TF-IDF de palavras/caracteres +
LinearSVC com `C=0.05`, incluindo vocabulários, IDF, coeficientes e interceptos.
Tem cerca de 3.59 MB e pode ser versionado em Git comum.

SHA-256 do artefato original:

```text
9a30abb664d5023d70f444641c4095cf800acba0c98fd8658087e7adad78ee12
```

`metadata.json`, `config.json` e `frozen_manifest.json` preservam procedência
e parâmetros EP1. `manifest.json` identifica as cópias distribuídas.
A inferência usa CPU sem retreinamento ou probabilidades artificiais.

Veja o [guia de uso](../../src/svm_classifier/README.md) e a
[avaliação teste1](../../docs/results/lexical-test1.md).
