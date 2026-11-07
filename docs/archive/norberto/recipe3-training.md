<a id="norberto-training"></a>

# Treinamento do NorBERTo

Fluxo histórico desativado em 2026-11-05. O modelo selecionado é o adaptador
original da entrega do EP1, sem alterações. Os comandos abaixo descrevem o
experimento anterior; sua implementação ativa de treino/avaliação foi removida.
Cópias locais do código estão em `results/norberto-rollback-20261105/sources-before-rollback.tar.gz`.

[Índice da documentação](../../README.md)

A receita 3 reproduz as unidades originais do otimizador do EP1 para limitação
e AdamW, normalizando os pesos dos exemplos entre microlotes acumulados.
Os comandos abaixo reproduzem essa receita; um substituto estável treinado no
corpus completo ainda não foi validado. O [guia do otimizador](recipe3-optimizer-units.md)
documenta a implementação, a regressão reintroduzida pela compatibilidade e a
verificação; receitas anteriores estão no
[arquivo de experimentos](../README.md#norberto-training-investigations).

A execução completa mais recente da receita 3 apresentou escalada transitória
dos gradientes e pior ajuste nas linhas compartilhadas de treino. A receita 3
corrige acumulação e unidades do otimizador; a recuperação no desenvolvimento
não estabelece estabilidade com mais dados. Veja a
[auditoria da escala de treino](../../results/README.md#training-scale-audit-recipe-3)
antes de planejar outra execução completa sem alterações. A auditoria não
validou nenhum hiperparâmetro corretivo.

<a id="environment"></a>

## Ambiente

Execute na raiz do EP2. O script exige fontes do EP1 vizinho, cache Hugging
Face e a imagem local validada `pln-rocm:6.4.1`. `NORBERTO_EP1_ROOT` e
`NORBERTO_IMAGE` selecionam outra cópia local ou imagem. O contêiner é offline,
monta EP1/cache somente para leitura e usa `HSA_OVERRIDE_GFX_VERSION=10.3.0`
na RX 6650 XT.

<a id="train-on-100-of-the-training-data"></a>

## Treinar em 100% dos dados de treinamento

```bash
NORBERTO_IMAGE=pln-rocm:6.4.1 \
bash scripts/norberto/train.sh \
  --partition full \
  --output-dir /workspace/results/norberto-full-recipe3
```

Inicializa um adaptador novo da base pública fixada e usa as **20,092 respostas
de treino do EP1**, incluindo o antigo conjunto reservado. Treina duas épocas
fixas sem validação nem seleção de checkpoints. O antigo conjunto reservado
é treino deste adaptador e não pode servir como validação inédita. Execuções
históricas completas levaram aproximadamente 39–40 minutos no ambiente registrado
da RX 6650 XT; a duração não garante conclusão.

<a id="train-on-the-frozen-development-partition"></a>

## Treinar na partição congelada de desenvolvimento

```bash
NORBERTO_IMAGE=pln-rocm:6.4.1 \
bash scripts/norberto/train.sh \
  --partition development \
  --output-dir /workspace/results/norberto-development-recipe3-new
```

Usa as **16,093 linhas originais de desenvolvimento** e reserva **3,999 linhas**,
sem nova divisão aleatória. Checksum da divisão, IDs, contagens, cobertura de
classes e ausência de sobreposição de linhas/grupos são verificados antes da
tokenização. A divisão padrão é
`EP1/results/research/dataset-hypotheses-20260923/splits.csv`; `--holdout-splits`
pode selecionar outra cópia com o mesmo checksum.

A execução concluída da receita 3 no desenvolvimento levou 31.32 minutos.
A qualidade é relatada separadamente na [seção de resultados](../../results/README.md),
com tabelas distintas para validação no conjunto reservado e comparação em teste1.

<a id="validate-the-plan-before-training"></a>

## Validar o plano antes de treinar

Acrescente `--dry-run` a qualquer comando. Esse modo verifica dados e parâmetros
sem importar bibliotecas dos modelos, criar o diretório de saída ou treinar:

```bash
bash scripts/norberto/train.sh \
  --partition full \
  --output-dir /workspace/results/norberto-full-recipe3-preview \
  --dry-run
```

Cada execução exige novo diretório de saída. Um diretório existente é rejeitado;
altere `--output-dir` em vez de sobrescrever artefatos anteriores.

<a id="fixed-parameters-and-provenance"></a>

## Parâmetros fixos e procedência

| Parâmetro | Valor |
|---|---|
| Base | `Itau-Unibanco/NorBERTo-base` |
| Revisão pública | `db73446f89c96044863ea05a39f680524b84bccb` |
| Épocas | 2 |
| LoRA | Posto 4, alpha 8, dropout 0.05, `Wqkv` nas 22 camadas |
| Contexto | 512 tokens, truncamento início/fim, maiúsculas/minúsculas preservadas |
| Precisão / atenção | FP32 / SDPA |
| Lote / acumulação | 4 / 4 |
| Taxa de aprendizado | 0.001 |
| Decaimento de pesos / proporção de aquecimento | 0.01 / 0.1 |
| Semente / threads em CPU | 42 / 2 |
| Receita | Versão 3, escala explícita 4 do gradiente do otimizador |

O invólucro reutiliza a função de treinamento do EP1 com a configuração efetiva
em `models/norberto/config.json` e a extensão Trainer da receita 3. Registra
hashes das fontes, dados/divisão, unidades do otimizador, linhas selecionadas,
estado de conclusão e checksums exportados. A perda reportada é a média por
exemplo. As normas registradas usam unidades do otimizador do EP1; divida por
4 ao comparar com as normas por amostra da receita 2.

<a id="outputs-and-evaluation"></a>

## Saídas e avaliação

Cada diretório de saída selecionado contém:

| Artefato | Conteúdo |
|---|---|
| `final_model/` | Novo adaptador LoRA, cabeça de classificação e tokenizador |
| `run.json`, `training.json`, `config.json` | Estado, receita, parâmetros, métricas e hashes |
| `training-rows.jsonl` | IDs das linhas de treino, rótulos e hashes dos textos |
| `checkpoints/` | Checkpoints intermediários de treinamento |

Um novo adaptador é distinto do modelo congelado do EP1 e não é instalado
automaticamente no pacote de inferência nem selecionado pelo ensemble.

Use o [fluxo de teste1 em duas etapas](../../guides/norberto-test1.md) para
predizer e depois avaliar as previsões salvas. Para uma execução concluída
somente de desenvolvimento, o script separado ``evaluate-holdout.py`` pode
validar o conjunto reservado congelado. Resultados e auditorias de treino estão
no [relatório da receita 3](../../results/norberto-recipe3-holdout.md).
