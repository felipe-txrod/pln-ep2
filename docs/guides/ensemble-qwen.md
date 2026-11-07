<a id="soft-voting-and-stacking-with-the-selected-qwen"></a>

# Votação suave e stacking com o Qwen selecionado

As duas famílias de ensemble agora suportam SVM congelado, RL somente de
desenvolvimento, NorBERTo original do EP1 e o Qwen selecionado em
`models/qwen/adapter/`. Usam a mesma partição supervisionada de 16,093 linhas e
conjunto reservado de 3,999. Os presets existentes de maioria permanecem
separados e mantêm três votantes.

<a id="selected-artifacts-and-data-roles"></a>

## Artefatos selecionados e papéis dos dados

| Método | Artefato padrão | Entradas |
|---|---|---|
| Votação suave calibrada | `models/ensemble-soft-qwen/model.json` | Quatro distribuições com pesos explícitos |
| Stacking de probabilidades | `models/ensemble-stacking-qwen/model.json` | Doze probabilidades calibradas |

Os papéis do conjunto reservado são idênticos em bytes aos dos ensembles
originais de três modelos: 2,027 linhas calibram probabilidades e 1,972
selecionam pesos da votação suave ou treinam o meta-modelo de stacking.
Linhas e componentes de similaridade são disjuntos.

O SVM usa calibração sigmoide. RL, NorBERTo e Qwen usam temperatura escalar
somente quando uma verificação agrupada de três folds melhora a perda
logarítmica nas linhas de calibração. A grade contém os 35 pontos distintos
do simplex de quatro membros, em incrementos de um quarto, incluindo pesos
uniformes de 25%. Macro-F1 escolhe a mistura; acurácia e distância aos pesos
uniformes desempatam. Pesos zero são permitidos e devem ser reportados.
O peso selecionado do Qwen é zero.

O stacking usa as 12 probabilidades, incluindo modelos com peso zero na votação
suave. Cinco folds agrupados nas 1,972 linhas de meta-validação selecionam
`C=[0.01, 0.1, 1, 10]` pela regra de um erro-padrão; C escolhido é 0.1.
Os coeficientes JSON exportados reproduzem as probabilidades da regressão
logística sem pickle nem ajuste durante a predição.

Qwen não recebeu atualizações de gradiente no conjunto reservado, mas esse
conjunto selecionou seu checkpoint. Isso e a exposição anterior em pesquisa
impedem tratar essas entradas como validação final inédita. Os rótulos de
Teste1 não calibram, selecionam pesos nem ajustam o meta-modelo.

<a id="prediction"></a>

## Predição

Com os arquivos registrados de escores de teste1 disponíveis, a combinação
usa somente CPU. Use novos diretórios:

```bash
PYTHONPATH=src python3 -m ensemble_classifier predict-soft \
  --input data/test1.xlsx --output-dir results/soft-qwen-test1-new

PYTHONPATH=src python3 -m ensemble_classifier predict-stacking \
  --input data/test1.xlsx --soft-run results/soft-qwen-test1-new \
  --output-dir results/stacking-qwen-test1-new
```

Para dados novos ou uma cópia do repositório sem os escores brutos ignorados,
use inferência em GPU e os caches das bases públicas selecionadas. Os modelos
neurais executam sequencialmente, liberando as alocações de GPU entre modelos:

```bash
PYTHONPATH=src python3 -m ensemble_classifier predict-soft \
  --input data/test2.xlsx --output-dir results/soft-qwen-test2-new \
  --device gpu --live-norberto --live-qwen

PYTHONPATH=src python3 -m ensemble_classifier predict-stacking \
  --input data/test2.xlsx --soft-run results/soft-qwen-test2-new \
  --output-dir results/stacking-qwen-test2-new
```

Execute a predição direta no ambiente ROCm existente com cache bitsandbytes e
mapeamentos de dispositivos GPU, conforme o [guia do Qwen](qwen-finetuning.md).
Esses comandos não treinam modelos-base, reajustam calibradores nem leem o
gabarito. Os exemplos de teste2 não foram executados nesta integração.

<a id="reproducing-the-fit"></a>

## Reprodução do ajuste

As configurações padrão identificam quatro membros e fixam os checksums do
arquivo de escores e da execução do Qwen. Reproduzir o ajuste exige esses
escores locais registrados:

```bash
PYTHONPATH=src python3 -m ensemble_classifier fit-soft \
  --config configs/ensemble-soft.json --output-dir results/soft-qwen-fit-new
```

Para ajustar outro artefato de stacking, copie `configs/ensemble-stacking.json`
e aponte `soft_model`, `soft_partition` e `holdout_scores` para o ajuste imutável
desejado. Não misture calibradores, partições ou escores de pacotes distintos.

Os modelos de três membros permanecem em `models/ensemble-soft/` e
`models/ensemble-stacking/`. Use `--model` para selecioná-los explicitamente.
As configurações originais estão preservadas em `configs/ensemble-soft-legacy.json`
e `configs/ensemble-stacking-legacy.json`. Somente os dois checksums originais
auditados recebem compatibilidade histórica; outros artefatos ainda exigem a
identidade exata da implementação.

O [relatório da integração](../results/ensemble-qwen.md) apresenta a comparação
medida em teste1 e a verificação independente.
