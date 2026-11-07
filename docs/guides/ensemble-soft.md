<a id="ensemble-por-weighted-soft-voting"></a>

# Ensemble por votação suave ponderada

Para gerar o Excel oficial com a configuração final 75/25, use
`bash scripts/ensemble/deliver-test2.sh`, conforme o
[guia de entrega nativa](final-delivery.md).

Este documento preserva o protocolo e os resultados originais de **três modelos**.
Os padrões atuais de quatro modelos incluem Qwen; veja o
[fluxo atual](ensemble-qwen.md) e os [resultados da integração](../results/ensemble-qwen.md).

O novo fluxo usa os modelos congelados e o formato `predictions.jsonl` do
pipeline existente. Os comandos de maioria/consenso e suas configurações
permanecem disponíveis. Não há treinamento de SVM, LR, NorBERTo ou TF-IDF.
Somente os pós-processadores de calibração e os pesos são ajustados.

## Modelos e separação dos dados

- SVM: `models/svm/`, LinearSVC C=0,05 do EP1, ajustado em 16.093 linhas.
- LR: `models/rl-development/`, C=0,25, já treinada na mesma partição de 16.093.
  A LR histórica de `models/rl/` e `EP1/artifacts/final-logistic-20260928.joblib`
  viu toda a base e não pode usar o holdout para ajustar o ensemble sem vazamento.
  A implementação reutiliza a alternativa de desenvolvimento **já existente**;
  não a retreina e não a apresenta como os mesmos bytes da LR histórica.
- NorBERTo: adapter original de entrega do EP1, duas épocas em 16.093 linhas,
  contexto 512. As probabilidades históricas podem ser importadas após conferir
  hashes, linhas, modelo, dataset e manifesto de origem.

O split original fixa 16.093 linhas de desenvolvimento e **3.999 de holdout**.
`StratifiedGroupKFold(2, shuffle=True, random_state=42)` divide somente esse
holdout em duas partes estratificadas por classe, mantendo juntos os componentes
de similaridade do EP1. Não se redefine o treinamento dos classificadores.

| Uso | Linhas | Grupos |
|---|---:|---:|
| Calibração | 2.027 | 1.627 |
| Escolha dos pesos | 1.972 | 1.654 |

Zero linhas/grupos compartilhados entre esses dois papéis. Também são
verificados o checksum do split original, ausência de grupos/textos normalizados
compartilhados com desenvolvimento e a lista de treinamento da LR. A partição
com IDs e hashes está versionada em `models/ensemble-soft/partition.jsonl`.

## Probabilidades e calibração

A ordem canônica é sempre **c1, c234, c5**. Dicionários são ordenados pelas
classes, arrays exigem ordem explícita, e valores ausentes, inválidos,
não finitos ou distribuições não normalizadas são rejeitados.

O `LinearSVC` existente **não tem `predict_proba()`**. Seus margens não são
probabilidades. Usa-se Platt/sigmoid one-vs-rest com normalização multiclasse,
por `CalibratedClassifierCV(FrozenEstimator(...), method="sigmoid")` sobre os
escores congelados, exclusivamente nas 2.027 linhas de calibração. O adaptador
de escores proíbe `.fit()`; o wrapper congelado impede treino do estimador.
Os parâmetros numéricos exportados são conferidos contra `predict_proba()`
do calibrador antes de serem salvos sem pickle.

LR fornece probabilidades nativas; NorBERTo fornece softmax. Para verificar a
necessidade de corrigir a escala, somente dentro da parte de calibração faz-se
uma checagem agrupada de três folds: identidade versus uma temperatura escalar,
limitada a [0,25; 10]. Temperatura só é adotada se reduzir log-loss fora da
amostra nessa checagem em mais de 1e-4; depois o único escalar é ajustado na
parte inteira de calibração. Essa etapa não usa rótulos da parte de pesos nem
do teste1. Temperatura positiva preserva o argmax individual.

Parâmetros congelados: T(LR)=1,1391917667 e T(NorBERTo)=1,1171232074. Na parte
de pesos, que não calibrou os modelos, log-loss antes/depois foi:

| Modelo | Antes | Depois |
|---|---:|---:|
| SVM | 1,047390 | 1,039346 |
| LR | 1,043068 | 1,042355 |
| NorBERTo | 1,061237 | 1,057509 |

Para SVM, “antes” é apenas o softmax diagnóstico das margens, **não** uma
probabilidade nativa ou um método de calibração. O relatório também guarda
Brier multiclasse, ECE e bins de confiabilidade. Distribuições normalizadas
e melhores escores probabilísticos não garantem calibração perfeita em dados
futuros; log-loss/Brier também refletem poder discriminativo.

## Escolha dos pesos

`p_ensemble(c|x) = soma_m peso_m * p_calibrado_m(c|x)`.

Uma única grade fixa: 15 pontos do simplex em incrementos de 0,25, mais pesos
iguais, total de **16 candidatos**. Os pesos são não negativos e somam um;
zero é permitido. Objetivo primário: F1-macro nas 1.972 linhas destinadas aos
pesos. Empate: maior acurácia, depois maior proximidade aos pesos iguais,
mantendo a ordem determinística da grade. Não há busca adaptativa ou aleatória.

Seleção congelada antes da avaliação final: **SVM 0,75; LR 0,00; NorBERTo 0,25**.
F1-macro de seleção: 47,5339%; acurácia: 47,5152%. Isso é desempenho de seleção,
não teste independente. Todos os três foram avaliados; a solução de fronteira
atribuiu zero à LR, portanto a mistura final tem dois contribuintes efetivos.
Não se alteram os pesos posteriormente para melhorar o teste1.

Não se chama `VotingClassifier.fit()`, que ajustaria clones dos estimadores.
A combinação é feita diretamente sobre probabilidades dos modelos existentes.
O modelo numérico, seus hashes, diagnóstico e grade estão em `models/ensemble-soft/`.

## Execução e avaliação separadas

Execute a partir de `EP2/`. O runtime lexical exige as versões em
`requirements-lexical.txt`, especialmente joblib 1.5.2. O ambiente do EP1
também pode ser reutilizado com `PYTHONPATH=src ../EP1/.venv/bin/python ...`.
Os comandos abaixo usam o ambiente virtual do EP2, após instalar o projeto.

```bash
.venv/bin/python -m pip install -r requirements-lexical.txt

# Reproduzir calibração/pesos, sem abrir os dados ou rótulos das 900 linhas:
.venv/bin/python -m ensemble_classifier fit-soft \
  --config configs/ensemble-soft-legacy.json --output-dir results/novo-soft-fit

# Aplicar o modelo já congelado; não lê clarity nem ajusta parâmetros:
.venv/bin/python -m ensemble_classifier predict-soft \
  --model models/ensemble-soft/model.json --input data/test1.xlsx \
  --output-dir results/novo-soft-test1

# Só agora abrir o gabarito do professor para avaliar os resultados salvos:
.venv/bin/python -m ensemble_classifier evaluate-soft \
  --run results/novo-soft-test1 \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/nova-avaliacao-soft-test1
```

O ajuste padrão requer o split e os CSVs preservados do EP1, indicados na
configuração e protegidos por checksums. Em outra máquina, forneça esses arquivos
ou use uma configuração com `recorded_norberto={}` para inferência local do
adapter verificado. Para predizer outras planilhas sem os CSVs históricos,
`predict-soft --live-norberto` usa o loader existente; requer a base restaurada
por `scripts/models/prepare-base.py` e as dependências opcionais NorBERTo.
`--device gpu` é explícito; por padrão usa CPU. Não treina nem baixa pesos durante
predição. A execução desta rodada reutilizou os CSVs, sem GPU ou inferência neural.

Os escores nativos são persistidos por membro para retomada. Alteração em dados,
pesos, classes, configuração ou código verificado é rejeitada; use outro
diretório e preserve os anteriores. O `evaluate-soft` confere e lê todas as
previsões nativas, calibradas e do ensemble antes de abrir o gabarito. Ele não
chama nenhuma função de ajuste. Exporta acurácia, F1-macro e matriz de confusão
para cada modelo nativo, cada modelo calibrado e a mistura final.

Esta comparação final do método nos 900 rótulos foi solicitada separadamente;
não é uma entrega de teste2 nem substitui o protocolo oficial teste2 → teste1.
O projeto já expôs anteriormente métricas de teste1 e do antigo holdout. Não se
alega teste historicamente cego; nesta implementação seus rótulos não escolhem
calibração ou pesos. Não modificar parâmetros após a avaliação final.

## Fontes verificadas

- [Calibração no scikit-learn 1.7](https://scikit-learn.org/1.7/modules/calibration.html).
- [CalibratedClassifierCV](https://scikit-learn.org/1.7/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).
- [FrozenEstimator](https://scikit-learn.org/1.7/modules/generated/sklearn.frozen.FrozenEstimator.html).
- [VotingClassifier](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.VotingClassifier.html).
