# Stacking dos três classificadores congelados

O comando `fit-stacking` treina somente uma nova Regressão Logística como
meta-modelo. SVM, LR de desenvolvimento, NorBERTo e seus calibradores são os
mesmos artefatos do [soft voting](ensemble-soft.md), sem retreinamento. A LR
histórica ajustada nas 20.092 linhas continua excluída: ela já viu o holdout.
Os presets de maioria e o weighted soft voting anterior não são substituídos.

## Separação dos dados

O conjunto real tem 20.092 linhas, divididas originalmente em 16.093 para os
classificadores e 3.999 de holdout. O stacking verifica os hashes dos modelos,
do corpus, do split e da partição publicada em `models/ensemble-soft/`.

| Uso | Linhas | Grupos de similaridade | c1 | c234 | c5 |
|---|---:|---:|---:|---:|---:|
| Calibração já congelada | 2.027 | 1.627 | 635 | 710 | 682 |
| Pesos do soft voting / meta-modelo | 1.972 | 1.654 | 622 | 707 | 643 |
| Avaliação final solicitada | 900 | Não define ajustes | 300 | 300 | 300 |

Os 1.627 grupos de calibração e os 1.654 do meta-modelo são disjuntos.
As 2.027 linhas não entram no ajuste nem na validação cruzada da meta-LR.
As 1.972 linhas não participaram de calibração. Os grupos originais também
não se sobrepõem aos de treinamento dos modelos-base.

Usar as mesmas 1.972 linhas que escolheram os pesos do soft voting preserva
a comparação: ambos combinam os mesmos modelos, calibradores e observações
de desenvolvimento. O stacking usa os três classificadores, incluindo a LR
que recebeu peso zero na mistura anterior. O soft voting permanece congelado
com SVM 0,75, LR 0 e NorBERTo 0,25; não se refaz sua busca para favorecer a
comparação. Não se compara desempenho de seleção dos pesos com OOF do stacking
como se fossem estimativas equivalentes: o teste1 é a comparação final comum.

É stacking de modelos pré-treinados com um conjunto disjunto para o meta-modelo.
Não é preciso gerar novas previsões OOF dos modelos-base: eles nunca treinaram
nas linhas do meta-modelo e os calibradores também não as viram. A CV aqui
treina somente a meta-LR. Usar `StackingClassifier.fit()` com comportamento
padrão retreinaria os modelos-base, contrariando o requisito; a combinação
explícita de escores permite reutilizar os loaders e os caches existentes.

## Entradas, meta-modelo e escolha de regularização

As nove entradas são probabilidades calibradas na ordem fixa:

```text
svm:c1, svm:c234, svm:c5,
rl:c1, rl:c234, rl:c5,
norberto:c1, norberto:c234, norberto:c5
```

Os leitores alinham rótulos explícitos e validam linhas, hashes de texto e
probabilidades normalizadas. SVM usa a sigmoid/Platt já ajustada; LR e
NorBERTo usam as temperaturas do soft voting. Não se misturam margens com
probabilidades. Não há texto bruto, rótulo verdadeiro ou identificador de linha
entre as entradas da meta-LR. Não se adiciona escala: todas as entradas estão
em [0,1]. A dependência linear entre probabilidades de uma mesma distribuição
é mantida; a penalidade L2 regulariza o ajuste.

Configuração definida antes de abrir o gabarito nesta rodada:

- LR multinomial com intercepto, penalidade L2, `solver=lbfgs`, sem pesos de classe.
- Grade fixa `C=[0.01, 0.1, 1, 10]`, sem busca adaptativa.
- Cinco folds de `StratifiedGroupKFold`, `shuffle=True`, semente 42, só nas 1.972 linhas.
- Objetivo: média não ponderada do F1-macro dos cinco folds.
- Regra de um erro-padrão: escolher o menor C cuja média esteja pelo menos em
  `melhor_média - erro_padrão_do_melhor`. Erro-padrão = desvio amostral / raiz de 5.
  Isso favorece maior regularização quando os resultados são próximos; não
  constitui um teste de significância, pois os ajustes dos folds compartilham dados.
- `max_iter=2000`, `tol=1e-8`; aviso de não convergência interrompe o ajuste.
- Após selecionar C, ajustar somente o meta-modelo nas 1.972 linhas completas.
  Não incorporar as linhas de calibração nem usar os rótulos do professor.

C escolhido: **0,1**. F1-macro médio dos folds: 46,08%; F1-macro das previsões
OOF reunidas: 46,20%. Essas métricas serviram à seleção; não são resultados
de teste independente. São 20 ajustes de CV e um ajuste final de apenas nove
entradas; a execução em CPU é curta e não exige GPU ou chamadas remotas.

`model.json` exporta coeficientes e interceptos em JSON, conferidos contra
`LogisticRegression.predict_proba()` a tolerância absoluta 1e-12. Inferência
usa softmax dos logits do meta-modelo, sem pickle ou `.fit()`. O manifesto
protege modelo, partição, folds, busca e previsões OOF. O artefato também fixa
o código e o checksum do soft voting que fornece os calibradores.

## Comandos reproduzíveis

Execute na raiz de EP2, com o ambiente já usado pelo soft voting. Use sempre
diretórios novos. Os escores das 3.999 linhas estão em cache; se faltarem, o
ajuste falha explicitamente, sem iniciar inferência neural silenciosamente.
Restaure o cache ou execute `fit-soft` separadamente para reconstruí-lo.

```bash
# Somente treino/seleção do meta-modelo; não abre teste1 nem teste2.
.venv/bin/python -m ensemble_classifier fit-stacking \
  --config configs/ensemble-stacking.json \
  --output-dir results/novo-stacking-fit

# Predição sem ler clarity, reutilizando os mesmos escores do teste1 já salvos.
.venv/bin/python -m ensemble_classifier predict-stacking \
  --model models/ensemble-stacking/model.json \
  --input data/test1.xlsx \
  --soft-run results/ensemble-soft-test1-20261105 \
  --output-dir results/novo-stacking-test1

# Avaliação separada: só agora lê o gabarito e não ajusta nenhum parâmetro.
.venv/bin/python -m ensemble_classifier evaluate-stacking \
  --run results/novo-stacking-test1 \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/nova-avaliacao-stacking
```

Para usar o modelo que acabou de ajustar, troque `--model` por
`results/novo-stacking-fit/model.json`. O caminho publicado acima usa o artefato
congelado distribuído. `configs/ensemble-stacking.json` aponta apenas para o
modelo soft, sua partição e seus escores de holdout, sem caminho de gabarito.

Para novos textos, omita `--soft-run`: o pipeline chama `predict-soft` em
`SAIDA/soft/`, usando os mesmos classificadores e calibradores, e aplica a
meta-LR. Para inferência neural local, use `--live-norberto --device gpu` e o
ambiente/dependências do [guia soft](ensemble-soft.md); por padrão usa CPU.
Esse caminho pode ser demorado se houver muitos textos inéditos. Se interromper,
retome o `predict-soft` na pasta interna e use-a com `--soft-run` em uma saída
nova do stacking. O comando de ajuste do meta-modelo nunca executa esse caminho.

A avaliação exporta acurácia, F1-macro, F1 por classe e matrizes dos dois
ensembles e dos três modelos nativos, além dos três calibrados. Compara os
acertos pareados por McNemar, como análise descritiva, e grava
`changes-vs-soft.jsonl` com linha Excel, hash, texto, rótulo verdadeiro,
previsões e categoria: `corrected`, `harmed` ou `both_wrong_changed_label`.
Os índices são as linhas reais da planilha, incluindo o cabeçalho na linha 1.
Não confundir mudança de uma classe errada para outra com correção.

## Limitações do experimento

O projeto já consultou historicamente o holdout e o teste1. Esta implementação
não usa o gabarito final para calibrar, escolher C ou pesos, mas a nova rodada
não transforma teste1 em um conjunto historicamente cego. O enunciado pede
teste2 antes de teste1; esta comparação isolada não certifica essa ordem.
Repetições herdadas entre os arquivos oficiais também limitam independência;
ver a [auditoria dos dados](../results/dataset-overlap.md). Não selecionar novos
parâmetros, modelos ou grades após observar a avaliação das 900 linhas.

## Documentação oficial consultada

- [StackingClassifier 1.7: modelos prefit e risco de reutilizar dados de treino](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.StackingClassifier.html).
- [LogisticRegression 1.7: L2, C e solução multinomial com lbfgs](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html).
- [StratifiedGroupKFold 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
