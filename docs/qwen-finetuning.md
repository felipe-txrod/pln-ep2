# Fine-tuning do Qwen3 na RX 6650 XT

## Visão geral

O experimento atual usa `Qwen/Qwen3-1.7B` para classificação supervisionada de
`resp_text` nas classes `c1`, `c234` e `c5`.
Não se usa GGUF nem se enviam textos da base para serviços de classificação.

A stack ROCm 6.4 instalada falhou na execução nativa `gfx1032`: o rocBLAS não
contém os kernels Tensile necessários. O lançador de contêiner usa
`HSA_OVERRIDE_GFX_VERSION=10.3.0` como workaround
por padrão. A GPU continua sendo fisicamente `gfx1032`; o override não equivale
a suporte oficial da AMD para essa combinação de GPU, Fedora e bibliotecas.

A otimização compara batches maiores e a retirada de recomputação mantendo
três épocas, contexto e batch efetivo. Assim, os ganhos de velocidade são
medidos com a mesma quantidade de treino.

## Ambiente e fontes de compatibilidade

Os testes utilizaram a imagem Docker local `pln-rocm:6.4.1`, identificador
iniciado por `1e951e93b719`. Python e bibliotecas executam no contêiner; o host
Fedora fornece o driver `amdgpu`, `/dev/kfd` e `/dev/dri`.

| Componente | Versão verificada |
| --- | --- |
| Host | Fedora 44; verificações iniciais no kernel 7.2.14 e finais no 7.2.15 |
| GPU | RX 6650 XT; 8176 MiB informados pelo PyTorch |
| Python | 3.12.3 |
| PyTorch / HIP | 2.8.0+rocm6.4 / 6.4.43482-0f2d60242 |
| Transformers / PEFT | 4.57.1 / 0.17.1 |
| Accelerate / bitsandbytes | 1.10.1 / 0.50.2 |
| Revisão do Qwen | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |

O Python 3.14 do host não contém essa stack. O ambiente de RAG do EP2 usa outras
versões de tokenizers e huggingface-hub; não instale as dependências do Qwen
nesse ambiente. `requirements-qwen.txt` fixa as bibliotecas do novo pipeline.

Fontes oficiais consultadas:

- [Modelo Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) e
  [configuração fixada](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/config.json).
- [Matriz do bitsandbytes 0.50.2](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation):
  as distribuições para ROCm 6.4.4 incluem `gfx1030`; outros builds mais novos
  listam `gfx1032`. Isso não garante execução na stack instalada: os testes
  numéricos locais continuam necessários.
- [Compatibilidade Radeon/Linux da AMD](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html).
- [Quantização com PEFT](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization).
- [Parâmetros do Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer):
  checkpointing troca memória por recomputação, e agrupamento por comprimento
  reduz padding quando o batch contém sequências de tamanhos diferentes.

## Instalação e contêiner

Execute a partir de `EP2/`. Para reutilizar o ambiente local já testado:

```bash
export QWEN_IMAGE=pln-rocm:6.4.1
export QWEN_RUNTIME_PATH=/workspace/.cache/qwen-runtime
```

O bitsandbytes foi instalado somente no cache ignorado do projeto, mantendo a
imagem original intacta. Em outro clone com a mesma imagem local, prepare-o com:

```bash
docker run --rm --pull=never --security-opt label=disable \
  -v "$PWD:/workspace" -w /workspace pln-rocm:6.4.1 \
  python3 -m pip install --no-deps --target /workspace/.cache/qwen-runtime \
  bitsandbytes==0.50.2
```

O lançador usa o UID/GID atual, monta o projeto e expõe os dispositivos AMD.
`QWEN_GFX_OVERRIDE=native` remove o workaround para diagnóstico da arquitetura
nativa.

`Dockerfile.qwen` oferece uma receita independente baseada em Ubuntu 24.04:

```bash
docker build -f Dockerfile.qwen -t ep2-qwen:rocm6.4 .
export QWEN_IMAGE=ep2-qwen:rocm6.4
unset QWEN_RUNTIME_PATH
```

Essa receita ainda não tem validação de execução registrada. Repita os testes de
compatibilidade ao construir outra imagem. A validação registrada corresponde
à imagem local e às versões indicadas acima.

A instalação nativa, em ambiente Python 3.12 separado, também exige instalar
primeiro o wheel ROCm, evitando resolver implicitamente um wheel CUDA:

```bash
python3.12 -m venv .venv-qwen
.venv-qwen/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/rocm6.4 'torch==2.8.0+rocm6.4'
.venv-qwen/bin/python -m pip install -r requirements-qwen.txt
.venv-qwen/bin/python -m pip install --no-deps -e .
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export AMDGPU_TARGETS=gfx1030
```

Esse caminho nativo de instalação não tem validação registrada. Instalar pacotes ou
identificar a GPU não substitui executar kernels reais.

## Reprodução e controle das execuções

Use um diretório novo para cada execução. O pipeline recusa sobrescrever
experimentos, adaptadores e entregas. O download é explícito; treino e inferência
carregam somente arquivos locais, com código remoto de modelos desabilitado.

Prepare os pesos públicos, o diagnóstico numérico e a partição agrupada:

```bash
bash scripts/qwen-container.sh download --config configs/qwen.json
bash scripts/qwen-container.sh probe --output results/qwen-probe-new.json
bash scripts/qwen-container.sh prepare --config configs/qwen.json \
  --input data/train_ep2.xlsx --output-dir results/qwen-prepared-new
```

Antes de um treino completo, o teste do modelo executa duas atualizações,
avaliação, salvamento dos adaptadores/cabeça e recarga com comparação de logits:

```bash
bash scripts/qwen-container.sh smoke --config configs/qwen-fast.json \
  --output-dir results/qwen-fast-smoke-new
```

Opcionalmente, verifique o fluxo com dados reais em um piloto de duas atualizações,
96 respostas de treino e 48 de validação:

```bash
bash scripts/qwen-container.sh train --pilot --config configs/qwen-fast.json \
  --smoke-report results/qwen-fast-smoke-new/summary.json \
  --output-dir results/qwen-fast-pilot-new
```

O comando de treino completo deve ser executado uma única vez, depois da
validação da configuração escolhida:

```bash
bash scripts/qwen-container.sh train --config configs/qwen-fast.json \
  --smoke-report results/qwen-fast-smoke-new/summary.json \
  --output-dir results/qwen-fast-training
```

Sem `--smoke-report`, o treino cria e executa automaticamente um teste adjacente
em `*-preflight`. Um relatório informado deve coincidir com configuração,
versões, ambiente, código e GPU. Mudanças nesses elementos invalidam relatórios
antigos. O treino repete testes numéricos na GPU exposta no momento da execução.

O diretório final `adapter/` contém pesos PEFT em safetensors, a cabeça treinada
`score`, tokenizer e `experiment.json`, com rótulos e revisão fixa da base.
A base original permanece necessária em `.cache/qwen-model/`; o adaptador não
é um modelo completo mesclado. Recalcule as métricas salvas com:

```bash
bash scripts/qwen-container.sh evaluate --run results/qwen-fast-training
```

## Dados, rótulos e avaliação

O treino contém 20.092 registros. Usa-se o primeiro fold de
`StratifiedGroupKFold(5, shuffle=True, random_state=42)`:

| Partição | c1 | c234 | c5 | Total |
| --- | ---: | ---: | ---: | ---: |
| Treino | 5039 | 5501 | 5470 | 16010 |
| Validação | 1308 | 1352 | 1422 | 4082 |

Os grupos usam HTML-unescape, espaços normalizados e lowercase, como no protocolo
anterior. Textos repetidos e rótulos conflitantes ficam na mesma partição. Não
se removem linhas de avaliação, não se corrigem rótulos e não se impõem cotas
para as previsões. O piloto faz subamostragem estratificada dentro de cada
partição; isso não cria compartilhamento de grupos entre treino e validação.

Somente `resp_text` é tokenizado. `clarity` vira o alvo inteiro `c1=0`, `c234=1`,
`c5=2`, sem entrar no texto. Acrescenta-se EOS após truncamento à direita;
padding e EOS têm IDs distintos, inclusive para respostas vazias. A inferência
ignora a coluna de rótulos mesmo em uma planilha rotulada. Preservam-se ordem,
IDs de linha e hashes dos textos e arquivos.

A leitura JSONL usa exclusivamente LF como delimitador: três separadores Unicode
presentes dentro de textos reais quebravam o uso anterior de `splitlines()`.
A correção preserva U+0085, U+2028 e U+2029 nas strings; há teste de regressão.

`validation-metrics.json` contém accuracy, macro-F1, macro-precision,
macro-recall e métricas/suporte por classe. A matriz de confusão tem linhas
verdadeiras e colunas previstas na ordem `c1`, `c234`, `c5`. Classes ausentes nas
previsões continuam participando. `validation-predictions.json` permite uma
verificação independente. A seleção de checkpoint usa acurácia de validação.

Depois de congelar o método, produza teste2 antes da comparação em teste1:

```bash
bash scripts/qwen-container.sh predict \
  --adapter results/qwen-fast-training/adapter --input data/test2.xlsx \
  --output-dir results/qwen-test2 --workbook deliveries/test2-qwen.xlsx

bash scripts/qwen-container.sh predict --phase statistical \
  --adapter results/qwen-fast-training/adapter \
  --input data/test1-rotulado-para-teste-estatistico.xlsx \
  --test2-run results/qwen-test2 --output-dir results/qwen-test1
```

Hashes oficiais de teste1/teste2 são recusados como fontes de treino. A inferência
oficial de teste1 exige uma execução concluída de teste2 com o mesmo adaptador.
A exposição histórica a teste1 deve ser declarada e não pode guiar a seleção.
A função existente `gpt_classifier.evaluation.compare` aceita a mesma estrutura
de linha/hash/rótulo para McNemar após completar o protocolo final.

## Resultados iniciais medidos

Os primeiros testes completos usaram batch 1, acumulação 16, contexto 256,
checkpointing e duas atualizações reais em dados sintéticos:

| Método | Pico alocado | Pico reservado | Recarga |
| --- | ---: | ---: | --- |
| QLoRA NF4 | 2476,71 MiB | 2922 MiB | Aprovada |
| LoRA FP16 | 3467,38 MiB | 3486 MiB | Aprovada |

Ambos treinam 3.217.408 parâmetros. O piloto real original obteve accuracy
0,375000, macro-F1 0,346341, macro-precision 0,408333 e macro-recall 0,373693 em
48 respostas. A matriz foi `[[7, 2, 6], [5, 2, 9], [8, 0, 9]]`.
Essas duas atualizações verificam integração; não estimam a qualidade final e
não permitem comparação direta com as avaliações anteriores de GPT ou EP1.

Artefatos: `results/qwen-smoke-verified/`,
`results/qwen-smoke-fp16-verified/`, `results/qwen-pilot-verified/` e
`results/qwen-pilot-inference/`. Na inferência separada, três textos sem rótulos
reproduziram as previsões salvas. Os agregados históricos estão em
[qwen-compatibility-results.json](qwen-compatibility-results.json).

## Otimização de memória e velocidade

A configuração inicial economizava VRAM com batch 1 e checkpointing. Os campos
`gradient_checkpointing`, `eval_batch_size` e `group_by_length` permitem controlar
recomputação, avaliação em lote e agrupamento por comprimento sem alterar a base,
os rótulos ou as partições. Batches e acumulação são comparados mantendo 16
exemplos por atualização; não confunda menos atualizações com mais velocidade.

O comando `benchmark` mede uma atualização de aquecimento e três cronometradas
com o contexto preenchido até o limite. Registra tempo por atualização, exemplos
por segundo, VRAM, finitude e atualização real dos adaptadores. OOM interrompe
os batches crescentes naquele processo, preservando os resultados anteriores.
Não rode benchmarks junto com um treino completo.

```bash
bash scripts/qwen-container.sh benchmark \
  --config results/qwen-performance/qlora-config.json \
  --batches 1 2 4 8 16 \
  --output-dir results/qwen-performance/qlora-no-checkpoint-new
```

O modo do benchmark precisa ser explícito (`qlora` ou `fp16`). Sem
`--checkpointing`, a recomputação fica desligada; use a opção para comparar.
A seleção deve usar o tempo por exemplo e manter margem para o desktop/driver,
não simplesmente escolher o maior batch ou preencher toda a VRAM.

### Comparação medida em 04/11/2026

Todos os testes abaixo processam 16 exemplos por atualização, com 256 tokens
por exemplo, um aquecimento e três atualizações cronometradas:

| Método | Batch | Checkpointing | Segundos/atualização | VRAM reservada |
| --- | ---: | --- | ---: | ---: |
| QLoRA NF4, baseline | 1 | Sim | 12,00 | 1786 MiB |
| QLoRA NF4 | 1 | Não | 8,18 | 2636 MiB |
| QLoRA NF4 | 2 | Não | 5,21 | 3786 MiB |
| QLoRA NF4 | 4 | Não | 4,07 | 6134 MiB |
| QLoRA NF4 | 8 | Não | OOM | 7742 MiB |
| QLoRA NF4 | 8 | Sim | 5,51 | 2490 MiB |
| QLoRA NF4 | 16 | Sim | 5,38 | 3532 MiB |
| LoRA FP16 | 2 | Não | 3,94 | 5242 MiB |
| LoRA FP16 | 4 | Não | 3,21 | 7076 MiB |

O melhor QLoRA medido é batch 4/acumulação 4, sem checkpointing: **2,95 vezes
mais throughput** que o baseline no mesmo teste, com 1607 MiB livres ao final.
Essa é a configuração de `configs/qwen-fast.json`, mantendo três épocas,
contexto 256, avaliação em batch 16 e agrupamento por comprimento.

FP16 batch 4 foi 21% mais rápido que esse QLoRA, mas deixou somente 676 MiB
livres no teste e muda o requisito original de quantização. Não se troca a
precisão do experimento principal sem registrar essa decisão. Os resultados
completos, incluindo falhas e tempos individuais, estão em
[qwen-performance-results.json](qwen-performance-results.json).

Esses números medem kernels e atualizações. A duração integral depende também
dos comprimentos reais, da avaliação, do carregamento e do comportamento térmico.
Os testes do modelo e o piloto validam o caminho de produção
depois da escolha; não estimam a qualidade final do classificador.

O modelo usa NF4 com dupla quantização, cálculo FP16, SDPA e KV cache desativado.
Somente LoRA e a cabeça recebem estados de otimizador. Embeddings congelados
ficam em FP16 após a preparação k-bit; parâmetros treináveis ficam em FP32.
AMP inicia com escala 128: a escala padrão causou overflow no teste original.
O callback verifica gradientes, atualização LoRA e descarte seguro de overflow;
falhas repetidas, loss ou logits não finitos impedem aceitar uma execução.

Na revisão da acumulação foi corrigida também a normalização da loss. No
Transformers 4.57.1, o forward de classificação do Qwen não repassa a contagem
de exemplos acumulados ao cálculo da entropia cruzada, embora aceite `**kwargs`.
O pipeline agora calcula explicitamente a soma das losses dividida pela
quantidade de exemplos da atualização completa. Isso cobre microbatches e
últimos lotes de tamanhos diferentes. Testes numéricos verificam equivalência
da loss, dos gradientes e de atualizações reais do Trainer em CPU.
Logs antigos de loss podem estar escalados pela acumulação e não devem ser
comparados diretamente com os novos. Os benchmarks já normalizavam corretamente;
os resultados de throughput permanecem válidos. A correção exige novo relatório
de teste, como qualquer alteração no código.

Picos do alocador PyTorch não incluem todo o uso do desktop/driver. Um teste
curto não certifica estabilidade térmica, convergência ou o término de épocas.
Mudanças em batch, contexto ou checkpointing precisam de novo teste do modelo.

## Alternativas e limitações

`mode=auto` verifica NF4 e o modelo completo antes de tentar FP16. Cada tentativa
usa outro subprocesso; uma falha HIP não contamina o fallback. Logs e falhas
são preservados. `mode=qlora` exige QLoRA e falha explicitamente se incompatível.

LoRA FP16 usa o mesmo caminho Transformers/PEFT e o mesmo override. Seus dois
passos sintéticos anteriores levaram 18,59 s, contra 25,58 s do NF4, mas a amostra
é curta. Mais memória disponível não garante maior velocidade nem substitui
benchmark. Se os dois métodos falharem, reduza batch/contexto ou use hardware
suportado. CPU FP32 não é um fallback implícito.

Checkpoints de época preservam otimizador, scheduler e RNG, mas a CLI inicial
não oferece retomada automática. Interromper antes do primeiro checkpoint pode
perder o progresso. Não sobrescreva o diretório interrompido nem troque arquivos
para fingir continuidade. Uma nova configuração usa outro destino e relatório.

### Validação final da configuração otimizada

O teste em `results/qwen-fast-final/` aprovou as duas atualizações, o salvamento
e a recarga após a correção da loss. O piloto real final em
`results/qwen-fast-final-pilot/` mediu média de **3,76 s por atualização**,
avaliação de 48 respostas em **5,47 s** e pico reservado de **6464 MiB**.
Não houve overflow. As métricas após apenas duas atualizações foram accuracy
0,395833, macro-F1 0,333892, macro-precision 0,489035 e macro-recall 0,407761;
elas verificam integração, sem demonstrar qualidade final.

O caminho principal usa QLoRA NF4. `configs/qwen-fp16-fast.json` contém a
alternativa FP16 medida em benchmark; essa configuração requer um novo teste
completo otimizado antes do treino.

Para treinar com a configuração otimizada e o relatório correspondente:

```bash
bash scripts/qwen-container.sh train --config configs/qwen-fast.json \
  --smoke-report results/qwen-fast-final/summary.json \
  --output-dir results/qwen-fast-training
```

Use esse novo relatório; os relatórios anteriores à correção da loss não
coincidem com o código atual. Preserve os diretórios de execuções anteriores.

## Código e testes

`src/qwen_classifier/` separa configuração, dados, diagnóstico de GPU, construção
PEFT, treinamento, inferência e benchmark.

A suíte final tem **116 testes aprovados**; antes da otimização eram 105.
Cobre rótulos,
agrupamento, linhas de planilha, JSONL Unicode, fallback, relatórios obsoletos e
proteções de teste1/teste2. Testes de GPU são execuções reais e separadas da suíte
simulada. A suíte completa também usa a entrega histórica do EP1 em
`../EP1/deliveries/test1.xlsx`, que deve ser montada somente para leitura no
contêiner. Nenhuma execução dessa suíte inicia classificações remotas.
