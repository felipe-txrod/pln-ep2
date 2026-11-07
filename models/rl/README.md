<a id="historical-full-corpus-logistic-regression"></a>

# Regressão logística histórica do corpus completo

Este artefato de 20,092 linhas é preservado para rastreabilidade e não é mais
a RL padrão do ensemble. O padrão é `models/rl-development/`, criado com
`bash scripts/rl/train.sh`; veja o
[fluxo somente de desenvolvimento](../../src/rl_classifier/README.md).

`model.joblib` contém o pipeline treinado TF-IDF de palavras/caracteres +
regressão logística com `C=0.25`, incluindo vocabulários, IDF, coeficientes e
interceptos. Tem cerca de 3.61 MB e pode ser versionado em Git comum.
O corpus final de treino continha 20,092 respostas.

SHA-256 do artefato original:

```text
f7af61142f43e1c8a37f0cd5a8191dd6d83ba1222cd8ecf12281f9929ac827c9
```

Configuração, metadados e manifestos acompanham os pesos. Inferência em CPU
retorna rótulos e probabilidades sem reajustar o classificador.

Veja o [guia de uso](../../src/rl_classifier/README.md) e a
[avaliação teste1](../../docs/results/lexical-test1.md).
