<a id="ep2--e-sic-clarity-classification"></a>

# EP2 — Classificação de clareza de respostas e-SIC

Classificação supervisionada de textos com Qwen3-1.7B, NorBERTo, SVM,
regressão logística e um ensemble configurável com arbitragem do GPT.
A entrada é `resp_text`; os rótulos de saída são `c1`, `c234` e `c5`.

A abordagem principal é **votação suave ponderada: SVM 75%, NorBERTo 25%**,
com **49.58% de macro-F1** no Teste1 (447/900 acertos; acurácia de 49.67%).
A [decisão adotada para o ensemble](docs/decisions/ensemble-strategy.md) registra
a justificativa, as limitações da avaliação e a preservação dos experimentos
com stacking, GPT e RAG.

O NorBERTo selecionado é o **adaptador original da entrega do EP1, sem alterações**,
em `models/norberto/`: LoRA com posto 4/alpha 8, duas épocas sobre 16,093 linhas
de desenvolvimento. Obteve 48.00% no teste1 e 45.31% no conjunto reservado do EP1.
As correções experimentais da perda, os retreinamentos e a variante de regressão
logística com codificador congelado foram desativados.

O ensemble majoritário usa **NorBERTo + SVM + RL**, com votos iguais e RL para
desempatar quando os três divergem. A votação suave e o stacking agora incluem
**Qwen** como quarto candidato/membro. Os quatro modelos reservam o mesmo
conjunto de 3,999 linhas do EP1.

O adaptador padrão do Qwen está em `models/qwen/adapter/`, treinado na mesma
partição 16,093/3,999 do EP1. Sua acurácia de validação é 44.79%, e a de teste1,
46.00%. A votação suave com quatro candidatos selecionou peso zero para Qwen;
o stacking utiliza suas probabilidades. Veja os [resultados da integração](docs/results/ensemble-qwen.md)
e a [verificação do modelo selecionado](docs/results/qwen-ep1-model.md).

## Gerar o Excel de entrega

Execução nativa, sem Docker. **CPU é o padrão do setup e da inferência**:

```bash
bash scripts/ensemble/setup-local.sh
bash scripts/ensemble/deliver-test2.sh
```

O setup prepara PyTorch CPU e as dependências neurais. Os perfis explícitos são:

| Uso | Comando de preparação |
|---|---|
| CPU completa (padrão) | `bash scripts/ensemble/setup-local.sh` |
| Somente base lexical e exportação de previsões existentes | `bash scripts/ensemble/setup-local.sh --minimal` |
| Bibliotecas neurais, preservando PyTorch já instalado no ambiente selecionado | `bash scripts/ensemble/setup-local.sh --with-neural` |
| Ambiente completo novo com PyTorch ROCm | `bash scripts/ensemble/setup-local.sh --with-rocm` |
| Ambiente completo novo com PyTorch CPU | `bash scripts/ensemble/setup-local.sh --with-cpu` |

`--without-rocm` mantém o comportamento mínimo de `--minimal`; não instala nem
substitui PyTorch. `--without-neural` omite as bibliotecas neurais. `--dry-run`
mostra os comandos sem instalar. Use `--venv` para indicar um ambiente existente.

A CPU é usada pelos modelos lexicais em todos os perfis. `--with-cpu` escolhe
uma distribuição do PyTorch sem GPU para o NorBERTo; não é uma opção para
habilitar o processador. O perfil mínimo não instala PyTorch e, sozinho, permite
apenas exportar previsões existentes. Para inferência, use um ambiente neural
completo, com PyTorch CPU ou ROCm.

Para usar GPU explicitamente, prepare ROCm e selecione o dispositivo:

```bash
bash scripts/ensemble/setup-local.sh --with-rocm
bash scripts/ensemble/deliver-test2.sh --device gpu
```

O arquivo final é **`deliveries/test2-soft-voting.xlsx`**, com as 897 respostas
classificadas pelo Soft Voting congelado 75/25. A geração aplica os modelos já treinados.
Cada execução recalcula as previsões de todos os membros e substitui os resultados
e o Excel anteriores após validação, sem reutilizar o cache de previsões.
Use `bash scripts/ensemble/deliver-test2.sh --check` para uma verificação sem
inferência. Exportação sem inferência e configuração do Python estão
no [guia da entrega final](docs/guides/final-delivery.md).

<a id="documentation"></a>

## Documentação

Comece pelo **[índice da documentação](docs/README.md)**. Guias operacionais,
resultados medidos, referências do enunciado e experimentos históricos estão separados.

| Tarefa | Guia |
|---|---|
| Reutilizar o adaptador original do NorBERTo do EP1 sem alterações | [Modelo original](src/norberto_classifier/README.md) |
| Predizer teste1 e depois comparar com o gabarito | [Fluxo do NorBERTo para teste1](docs/guides/norberto-test1.md) |
| Configurar, treinar e executar Qwen com QLoRA | [Ajuste fino do Qwen](docs/guides/qwen-finetuning.md) |
| Retreinar Qwen na mesma partição 16,093/3,999 dos modelos do EP1 | [Partição exata do EP1](docs/guides/qwen-finetuning.md#retrain-with-the-exact-ep1-partition) |
| Treinar regressão logística lexical somente no desenvolvimento, preservando o conjunto reservado | [Fluxo da RL](src/rl_classifier/README.md) |
| Selecionar os membros do ensemble e o modelo de desempate | [Ensemble](docs/guides/ensemble.md) |
| Executar votação suave calibrada e stacking de quatro modelos com Qwen | [Fluxo atual dos ensembles](docs/guides/ensemble-qwen.md) |
| Encaminhar decisões incertas do stacking ao GPT com validação OOF e cache persistente | [Cascata seletiva com GPT](docs/guides/stacking-gpt-cascade.md) |
| Preparar RAG somente no treino e quatro ablações pareadas de contexto; classificação mediante ativação explícita | [Estudo de contexto RAG + stacking](docs/guides/rag-stacking.md) |
| Comparar recuperação pelo NorBERTo congelado, MMR e aceitação de respostas GPT em cache, sem chamadas remotas | [Pesquisa offline](docs/guides/rag-offline-research.md) e [resultados medidos](docs/results/rag-offline-research.md) |
| Examinar o protocolo preservado de calibração de três modelos e meta-RL | [Votação suave](docs/guides/ensemble-soft.md) e [stacking](docs/guides/ensemble-stacking.md) |
| Consultar scripts executáveis por modelo e finalidade | [Índice dos scripts](scripts/README.md) |
| Restaurar modelos congelados e suas bases públicas fixadas | [Artefatos dos modelos](models/README.md) |
| Comparar resultados medidos dos classificadores | [Índice dos resultados](docs/results/README.md) |
| Reproduzir a votação majoritária NorBERTo + SVM + RL e sua comparação em teste1 | [Ensemble majoritário](docs/results/ensemble-majority.md) |
| Comparar votação suave ponderada calibrada com cada classificador congelado | [Resultados da votação suave e matrizes de confusão](docs/results/ensemble-soft.md) |
| Comparar ensembles de quatro modelos com os controles de três modelos | [Resultados da integração do Qwen](docs/results/ensemble-qwen.md) |
| Examinar a comparação original de stacking, incluindo as instâncias afetadas | [Resultados do stacking de três modelos](docs/results/ensemble-stacking.md) |
| Consultar estudos anteriores com GPT/Codex e investigações de treinamento | [Arquivo de experimentos](docs/archive/README.md) |

<a id="repository-layout"></a>

## Estrutura do repositório

```text
configs/       Classifier and experiment configurations
src/           Classifier packages and shared utilities
scripts/       Commands grouped by model, artifacts, diagnostics, and experiments
models/        Frozen model packages and base references
data/         Input workbooks
results/       Local runs and predictions; ignored by Git
docs/          Guides, results, reference material, and archive
```

<a id="base-installation"></a>

## Instalação básica

Use Python 3.11 ou posterior e execute os comandos na raiz do repositório:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
```

O treinamento e a inferência em GPU usam o ambiente ROCm documentado nos guias
dos modelos. Na RX 6650 XT, o contêiner validado é `pln-rocm:6.4.1`, com
`HSA_OVERRIDE_GFX_VERSION=10.3.0`. A instalação básica acima não seleciona
nem instala uma distribuição do PyTorch específica para GPU.

<a id="classifier-packages"></a>

## Pacotes dos classificadores

| Pacote | Função |
|---|---|
| `qwen_classifier` | Classificação com Qwen3-1.7B e adaptadores QLoRA |
| [norberto_classifier](src/norberto_classifier/README.md) | Adaptador original da entrega do EP1 sem alterações; somente inferência |
| [svm_classifier](src/svm_classifier/README.md) | Pipeline congelado de TF-IDF e SVM, inferência em CPU |
| [rl_classifier](src/rl_classifier/README.md) | Pipeline congelado de TF-IDF e regressão logística, inferência em CPU |
| [ensemble_classifier](src/ensemble_classifier/README.md) | Membros configuráveis e arbitragem sobre previsões salvas |
| `cascade_classifier` | Validação aninhada OOF do roteamento e decisões seletivas do GPT sobre stacking congelado |
| `rag_cascade` | Piloto de recuperação/contexto somente no treino e inferência com ordem obrigatória teste2 antes de teste1 |
| `rag_research` | Ablações offline de recuperação com representações congeladas e reavaliação de intervenções históricas do GPT |
| `gpt_classifier` | Classificação remota, recuperação de exemplos e avaliação compartilhada |

Os pacotes congelados ficam em `models/`; os adaptadores experimentais históricos
permanecem nos diretórios originais das execuções. O [guia dos modelos](models/README.md)
explica formatos, verificações de integridade, restauração e revisões das bases públicas.

<a id="data-and-evaluation"></a>

## Dados e avaliação

A planilha canônica de treino, `data/train.xlsx`, contém **20,092 respostas**.
A partição congelada do NorBERTo utiliza **16,093 linhas de desenvolvimento** e
**3,999 linhas reservadas**; treinar em 100% inclui esse antigo conjunto reservado.
Cada experimento registra sua própria partição.

**Teste1 tem 900 linhas** e serve à comparação final com o EP1. Gere as previsões
dos textos antes de compará-las ao gabarito. **Teste2 tem 897 linhas** e é a planilha
de entrega. Não selecione prompts, hiperparâmetros ou regras do ensemble usando
os rótulos de teste1. Os resultados do conjunto reservado e de teste1 descrevem
bases diferentes.

O [enunciado](docs/reference/assignment.md) e o
[modelo de relatório](docs/reference/report-template.md) definem os requisitos
acadêmicos. As métricas registradas e sua procedência estão na
[seção de resultados](docs/results/README.md).

<a id="verification"></a>

## Verificação

```bash
.venv/bin/python -m pytest -q
```

Dependências opcionais dos modelos e verificações reais de integração com GPU
estão nos respectivos guias. Comandos anteriores do GPT/Codex, estudos de
recuperação e registros de verificação estão preservados no
[arquivo](docs/archive/README.md).
