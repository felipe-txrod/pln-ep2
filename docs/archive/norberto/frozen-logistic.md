<a id="frozen-norberto-with-logistic-regression"></a>

# NorBERTo congelado com regressão logística

Fluxo histórico desativado em 2026-11-05. O modelo selecionado é o adaptador
original da entrega do EP1, sem alterações. Os comandos abaixo descrevem o
experimento anterior; sua implementação ativa de treino/avaliação foi removida.
Cópias locais do código estão em `results/norberto-rollback-20261105/sources-before-rollback.tar.gz`.

[Índice da documentação](../../README.md)

Este candidato usa a revisão pública fixada de `Itau-Unibanco/NorBERTo-base`
`db73446f89c96044863ea05a39f680524b84bccb` como extrator imutável de atributos.
Nenhum adaptador LoRA ou cabeça de classificação é carregado. A extração usa
modo de avaliação, execução somente de inferência, FP32 e nenhum parâmetro
treinável do codificador. Apenas uma regressão logística multinomial com
regularização L2 é ajustada.

<a id="representation"></a>

## Representação

A limpeza preserva maiúsculas/minúsculas e usa a tokenização início/fim
compatível com o EP1, com contexto de 512 tokens. A média do último estado
oculto ponderada pela máscara de atenção inclui tokens especiais atendidos e
exclui preenchimento. Cada vetor de 768 dimensões é normalizado em L2
individualmente. Não há escalonamento ajustado ao corpus nem agregação aprendida;
assim, os vetores fixos podem ser armazenados em cache e reutilizados entre
folds sem ajuste nas observações de validação.

O codificador verifica os SHA-256 da base pública/tokenizador. O hash do estado
carregado deve ser idêntico antes e depois da extração. O cache SQLite guarda
checksums por vetor e chaves que abrangem texto, identidade do codificador,
precisão, versões do ambiente, dispositivo, tamanho do lote e versão da extração.
Lotes concluídos são persistidos imediatamente; repita uma extração que falhou
com outro diretório de saída e o mesmo cache para reutilizá-los.

<a id="exact-ep1-norberto-validation-partitions"></a>

## Partições exatas de validação do NorBERTo do EP1

A implementação reproduz o protocolo original de avaliação do NorBERTo-512,
não o protocolo distinto de busca em grade aninhada de alguns estudos lexicais:

- A planilha inalterada do EP1 e a divisão congelada por componentes de
  similaridade têm hashes fixados.
- Apenas 16,093 linhas de desenvolvimento entram na extração, ajuste e avaliação
  OOF. As 3,999 linhas reservadas são excluídas antes da tokenização.
- Três folds externos estratificados por grupo usam semente 42. Dentro de cada
  treino externo, a primeira divisão de cinco folds estratificados por grupo usa
  semente `42 + fold`. Cada ID reconstruído é conferido com os arquivos do EP1
  de `split-1.json` a `split-3.json`.
- A grade predefinida de `C` é `1, 10, 100`, dos experimentos semântico-logísticos
  congelados do EP1. A acurácia interna seleciona `C`; empates preservam a ordem
  da grade. O classificador selecionado é avaliado no fold externo sem novo
  ajuste no treino externo, preservando o número de exemplos usado pelo
  codificador do EP1.
- O classificador final usa todas as linhas de desenvolvimento e a mediana dos
  três `C` escolhidos internamente. Resultados externos nunca selecionam
  parâmetros finais.

| Fold | Treino interno | Validação interna | Validação externa |
|---|---:|---:|---:|
| 1 | 8,572 | 2,199 | 5,322 |
| 2 | 8,769 | 2,176 | 5,148 |
| 3 | 8,151 | 2,319 | 5,623 |

O solucionador segue a família logística do EP1: L-BFGS, L2, máximo de 1,500
iterações, tolerância 0.0001 e sem pesos de classe. Não convergência gera erro,
em vez de exportar um classificador bem-sucedido. A configuração está em
``configs/norberto-frozen-logistic.json``.

<a id="commands"></a>

## Comandos

Execute na raiz do EP2. O script usa a imagem offline existente
`pln-rocm:6.4.1` e monta o EP1 vizinho somente para leitura. `NORBERTO_IMAGE` e
`NORBERTO_EP1_ROOT` substituem esses locais. Extração/inferência expõem a GPU;
`--device gpu` a seleciona explicitamente. Planejamento, validação cruzada e
ajuste final executam em CPU sem expor o dispositivo GPU.

```bash
bash scripts/norberto/frozen-logistic.sh plan

bash scripts/norberto/frozen-logistic.sh encode --device gpu \
  --output-dir results/norberto-frozen-logistic-new/embeddings

bash scripts/norberto/frozen-logistic.sh cross-validate \
  --features-dir results/norberto-frozen-logistic-new/embeddings \
  --output-dir results/norberto-frozen-logistic-new/cv

bash scripts/norberto/frozen-logistic.sh fit \
  --features-dir results/norberto-frozen-logistic-new/embeddings \
  --cv-dir results/norberto-frozen-logistic-new/cv \
  --output-dir results/norberto-frozen-logistic-new/model
```

Todo diretório de saída deve ser novo. O cache compartilhado padrão é
`.cache/norberto-logistic/embeddings.sqlite3`. `--base-dir`, `--cache` e
`--batch-size` selecionam outros locais/parâmetros explícitos. Python nativo usa
`PYTHONPATH=src python3 -m norberto_logistic`; a instalação editável também expõe
`norberto-logistic-ep2`. Use as dependências fixadas do projeto e um ambiente
PyTorch adequado, como no ambiente existente do NorBERTo.

O candidato local concluído está em
`results/norberto-frozen-logistic-ep1-20261105/model`. Para predizer com um
modelo escolhido explicitamente, por exemplo:

```bash
bash scripts/norberto/frozen-logistic.sh predict --device gpu \
  --model-dir results/norberto-frozen-logistic-ep1-20261105/model \
  --input data/test1.xlsx \
  --output-dir results/norberto-frozen-logistic-test1-new
```

A predição lê somente `resp_text` e grava IDs originais do Excel, hashes dos
textos, rótulos e probabilidades em `predictions.jsonl`. Não lê gabaritos,
treina classificadores nem avalia testes oficiais. Fora da extração/validação
cruzada de desenvolvimento, a verificação inicial usou três respostas sintéticas.
A avaliação posterior do conjunto reservado é descrita abaixo. Este candidato
não foi avaliado em nenhum dos testes oficiais.

<a id="holdout-evaluation"></a>

## Avaliação do conjunto reservado

No mesmo ambiente Python/ROCm fixado, avalie o classificador de desenvolvimento
salvo sem novo ajuste:

```bash
PYTHONPATH=src python3 scripts/norberto/evaluate-frozen-holdout.py \
  --training-run results/norberto-frozen-logistic-ep1-20261105 \
  --ep1-root ../EP1 --device gpu \
  --output-dir results/norberto-frozen-logistic-holdout-new \
  --recipe3-predictions results/norberto-recipe3-holdout-20261105/holdout/predictions.jsonl
```

A comparação com a receita 3 é opcional. O script exige novo diretório de saída
e valida a planilha/divisão fixadas, identidades das linhas de atributos de
desenvolvimento, hashes de modelo/validação cruzada, convergência, escolha de C
pela mediana somente interna e separação de linhas/grupos. Salva as 3,999
previsões geradas somente dos textos antes de construir registros rotulados;
depois avalia e alinha controles históricos salvos de LoRA, regressão logística
e SVM. Nunca chama ajuste de classificadores. Atributos, previsões, métricas,
estatísticas do codificador, hashes inalterados e comparações pareadas por
bootstrap de grupos de similaridade são registrados separadamente do treino.

A avaliação concluída está em
`results/norberto-frozen-logistic-holdout-20261105/holdout/`: 1,694/3,999 acertos,
42.36% de acurácia e 42.08% de macro-F1. Leitura independente da fonte e
reconstrução direta por NumPy verificaram todas as probabilidades e métricas.
O LoRA original atingiu 45.31% nessas mesmas linhas. Nenhum parâmetro mudou
após os resultados; o modelo não foi promovido.

<a id="artifacts-and-measured-quality"></a>

## Artefatos e qualidade medida

A extração exporta `features.npy` verificado, identidades de textos/linhas e
manifesto do codificador. A validação cruzada exporta três arquivos seguros das
cabeças lineares, métricas dos folds/tentativas internas, 16,093 previsões OOF,
hashes das fontes e comparações históricas pareadas. O ajuste final exporta
`classifier.npz` e `manifest.json`. O NPZ contém apenas coeficientes, interceptos
e ordem dos rótulos; o carregamento usa `allow_pickle=False`. A inferência
verifica as identidades do classificador e do codificador antes de predizer.

O [índice dos resultados](../../results/README.md#frozen-norberto-logistic-ep1-cross-validation)
registra OOF de desenvolvimento e a avaliação separadamente solicitada do
conjunto reservado, ambas com verificação independente. O codificador congelado
remove sua otimização deste pipeline; convexidade não garante melhor acurácia
nem melhora monotônica com mais dados. Este candidato permanece separado do
modelo publicado do EP1 e da seleção do ensemble.
